import importlib
import inspect
import os
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

from . import api_model_discovery
from .base import BaseAnnotator
from .config import AVAILABLE_API_MODELS_CONFIG_PATH, config_registry, load_available_api_models
from .types import AnnotatorInfo, ModelType, TaskCapability
from .utils import logger

T = TypeVar("T", bound=BaseAnnotator)
ModelClass = type[BaseAnnotator]

_MODEL_CLASS_OBJ_REGISTRY: dict[str, ModelClass] = {}
_REGISTRY_INITIALIZED: bool = False
_WEBAPI_MODEL_METADATA: dict[str, dict[str, Any]] = {}


# --- プライベートヘルパー関数 ---


def _list_module_files(directory: str) -> list[Path]:
    """指定されたディレクトリ内の全てのPythonモジュールファイル(サブディレクトリ含む、__init__.py除く)をリストアップ"""
    try:
        base_dir = Path(__file__).parent.parent
        abs_path = (base_dir / directory).resolve()
        if not abs_path.is_dir():
            logger.warning(f"モジュールディレクトリが見つかりません: {abs_path}")
            return []
        # 再帰的に.pyファイルを取得
        module_files = [p for p in abs_path.rglob("*.py") if p.name != "__init__.py"]
        logger.debug(f"{abs_path} 以下で {len(module_files)} 個のモジュールファイルが見つかりました")
        return module_files
    except Exception as e:
        logger.error(f"{directory} 内のモジュールファイルのリストアップ中にエラー: {e}", exc_info=True)
        return []


def _import_module_from_file(module_file: Path, base_module_path: str) -> ModuleType | None:
    # base_dirはmodel_classディレクトリ
    base_dir = Path(__file__).parent.parent / "model_class"
    rel_path = module_file.relative_to(base_dir)
    # 拡張子を除去し、パス区切りをドットに変換
    module_path = rel_path.with_suffix("").as_posix().replace("/", ".")
    full_module_path = f"{base_module_path}.{module_path}"
    try:
        module = importlib.import_module(full_module_path)
        logger.debug(f"モジュールのインポート成功: {full_module_path}")
        return module
    except ImportError as e:
        logger.error(f"モジュール {full_module_path} のインポート中にエラー: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"モジュール {full_module_path} のインポート中に予期せぬエラー: {e}", exc_info=True)
        return None


def _recursive_subclasses[T: BaseAnnotator](cls: type[T]) -> set[type[T]]:
    """指定されたクラスのすべての再帰的サブクラスを返します。"""
    # この関数はそのままですが、他で使用する場合はTが適切にバインドされていることを確認してください
    # _gather_available_classesでは、BaseAnnotatorを直接使用します。
    subclasses: set[type[T]] = set()
    try:
        # より良い互換性のためにinspect.getmro()を使用し、__subclasses__をチェック
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            subclasses.update(_recursive_subclasses(subclass))  # type: ignore
    except TypeError as e:
        logger.warning(f"{cls.__name__} のサブクラスを取得できませんでした (C拡張の可能性があります): {e}")
    except Exception as e:
        logger.error(f"{cls.__name__} のサブクラス検索中にエラー: {e}", exc_info=True)
    return subclasses


def _gather_available_classes(directory: str) -> dict[str, ModelClass]:
    """指定ディレクトリ内の全モジュールから、BaseAnnotator のサブクラスまたは
    predict() メソッドを持つクラスを抽出して返す"""
    available: dict[str, ModelClass] = {}
    module_files = _list_module_files(directory)
    for module_file in module_files:
        # モジュールインポートパスを修正 (ユーザー変更を反映)
        module = _import_module_from_file(module_file, "image_annotator_lib.model_class")
        if module is None:
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # 古いプロバイダー固有のクラスは除外(PydanticAI統一後は不要)
            obsolete_classes = [
                "AnthropicApiAnnotator",
                "GoogleApiAnnotator",
                "OpenAIApiChatAnnotator",
                "OpenAIApiResponseAnnotator",
            ]
            if name in obsolete_classes:
                logger.debug(
                    f"古いプロバイダー固有クラス '{name}' をスキップします(PydanticAI統一後は不要)"
                )
                continue

            # BaseAnnotator のサブクラスか、または predict メソッドを持つ場合 (ユーザー変更を反映)
            # objがtypeであることを確認 (Mypy対策)
            if isinstance(obj, type) and (
                (issubclass(obj, BaseAnnotator) and obj is not BaseAnnotator) or hasattr(obj, "predict")
            ):
                # Mypyエラー回避のため issubclass チェックを追加
                if issubclass(obj, BaseAnnotator):
                    available[name] = obj
                    # 再帰的サブクラス収集 (ユーザー変更を反映)
                    for subcls in _recursive_subclasses(obj):
                        # 古いクラスは再帰的サブクラスからも除外
                        if subcls.__name__ not in obsolete_classes:
                            available[subcls.__name__] = subcls
                elif hasattr(
                    obj, "predict"
                ):  # predictを持つがBaseAnnotatorサブクラスでない場合も登録 (元のロジック踏襲)
                    # このケースでは型が合わない可能性があるため注意が必要だが、元のロジックを維持
                    # 必要であれば、より厳密な型チェックや別の扱いを検討
                    available[name] = obj  # type: ignore

    logger.debug(f"{directory} から利用可能なクラス: {list(available.keys())}")
    return available


def _is_obsolete_annotator_class(class_name: str) -> bool:
    """古いプロバイダー固有のアノテータークラスかどうかを判定する。

    PydanticAI統合後、個別プロバイダークラス(例: OpenAIApiAnnotator,
    OpenAIApiChatAnnotator, OpenAIApiResponseAnnotator)は廃止。

    Args:
        class_name: チェック対象のクラス名。

    Returns:
        廃止クラスの場合True。
    """
    if class_name == "PydanticAIWebAPIAnnotator":
        return False
    # *ApiAnnotator, *ApiChatAnnotator, *ApiResponseAnnotator を検出
    obsolete_suffixes = ("ApiAnnotator", "ApiChatAnnotator", "ApiResponseAnnotator")
    return any(class_name.endswith(suffix) for suffix in obsolete_suffixes)


def _resolve_model_class(
    desired_class_name: str,
    model_name: str,
    available_classes: dict[str, type],
    pydantic_ai_class: type | None,
    model_type_name: str,
) -> type | None:
    """設定エントリからモデルクラスを解決する。

    WebAPIクラスは統一PydanticAI実装にマッピングし、
    古いプロバイダー固有クラスは警告してスキップする。

    Args:
        desired_class_name: 設定で指定されたクラス名。
        model_name: ログ用のモデル名。
        available_classes: スキャン済みの利用可能なクラス辞書。
        pydantic_ai_class: PydanticAIWebAPIAnnotatorクラス(なければNone)。
        model_type_name: ログ用のモデルタイプ名。

    Returns:
        解決されたモデルクラス、またはスキップすべき場合はNone。
    """
    # WebAPIクラス(PydanticAIWebAPIAnnotator)の場合は統一実装を使用
    if desired_class_name == "PydanticAIWebAPIAnnotator" and pydantic_ai_class:
        logger.debug(f"WebAPIモデル '{model_name}' にPydanticAI統一実装を使用")
        return pydantic_ai_class

    # 古いプロバイダー固有クラスは警告してスキップ
    if _is_obsolete_annotator_class(desired_class_name):
        logger.warning(
            f"モデル '{model_name}' で古いプロバイダー固有クラス '{desired_class_name}' が指定されています。"
            f"PydanticAI統合後はすべてのWebAPIモデルで 'PydanticAIWebAPIAnnotator' を使用してください。スキップします。"
        )
        return None

    # 非WebAPIクラス(ローカルMLモデルなど)は従来通りの処理
    model_cls = available_classes.get(desired_class_name)
    if model_cls is None:
        logger.warning(
            f"{model_type_name} モデル '{model_name}' で指定されたクラス '{desired_class_name}' が見つかりません。"
        )
    return model_cls


def _try_register_model(
    registry: dict[str, ModelClass],
    model_name: str,
    model_cls: type,
    base_class: type,
) -> bool:
    """単一モデルをレジストリに登録する。

    Args:
        registry: 登録先レジストリ。
        model_name: モデル名。
        model_cls: モデルクラス。
        base_class: 期待される基底クラス。

    Returns:
        登録に成功した場合True。
    """
    if not (issubclass(model_cls, base_class) or hasattr(model_cls, "predict")):
        logger.error(
            f"モデル '{model_name}' のクラス '{model_cls.__name__}' が "
            f"{base_class.__name__} を継承しておらず predict メソッドも持ちません。スキップします。"
        )
        return False

    if model_name in registry:
        if registry[model_name] is model_cls:
            logger.debug(f"モデル名 '{model_name}' は既に登録されています（同一クラス）。スキップします。")
            return True
        logger.warning(
            f"モデル名 '{model_name}' は既に登録されています。クラス '{model_cls.__name__}' で上書きします。"
        )
    registry[model_name] = model_cls
    logger.debug(f"モデル '{model_name}' をクラス '{model_cls.__name__}' でレジストリに登録しました。")
    return True


def _register_models(
    registry: dict[str, ModelClass],
    model_type_name: str,
    directory: str,
    base_module_path: str,
    base_class: type,
) -> None:
    """設定に基づいてモデルを登録する汎用関数。

    Args:
        registry: 登録先レジストリ辞書。
        model_type_name: ログ用のモデルタイプ名 (例: "annotator")。
        directory: モデルクラスファイルを含むディレクトリ (例: "model_class")。
        base_module_path: 基本Pythonインポートパス (例: "image_annotator_lib.model_class")。
        base_class: モデルが継承すべき基底クラス。
    """
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back if current_frame else None
    caller_info = inspect.getframeinfo(caller_frame) if caller_frame else None
    caller_repr = (
        f"{caller_info.filename}:{caller_info.lineno} in {caller_info.function}"
        if caller_info
        else "不明な呼び出し元"
    )

    logger.debug(f"{model_type_name} モデルの登録を開始します。呼び出し元: {caller_repr}")
    logger.debug(f"現在のレジストリの状態: {list(registry.keys())}")

    registered_count = 0

    try:
        config = config_registry.get_all_config()
        if not config:
            logger.warning("モデル設定が空か、ロードに失敗しました。モデルは登録されません。")
            return
        logger.debug(f"モデル設定をロードしました: 合計 {len(config)} エントリ。")

        available_classes = _gather_available_classes(directory)
        logger.debug(f"{len(available_classes)} 個の利用可能な {model_type_name} クラスが見つかりました")

        pydantic_ai_class = available_classes.get("PydanticAIWebAPIAnnotator")
        if not pydantic_ai_class:
            logger.error("PydanticAIWebAPIAnnotator クラスが見つかりません。WebAPIモデルは登録できません。")

        for model_name, model_config in config.items():
            desired_class_name = model_config.get("class")
            if not desired_class_name:
                logger.warning(
                    f"設定でモデル '{model_name}' のクラス名が指定されていません。スキップします。"
                )
                continue

            model_cls = _resolve_model_class(
                desired_class_name, model_name, available_classes, pydantic_ai_class, model_type_name
            )
            if model_cls is None:
                continue

            if _try_register_model(registry, model_name, model_cls, base_class):
                registered_count += 1

    except Exception as e:
        logger.error(
            f"予期せぬエラーのため {model_type_name} モデルの登録に失敗しました: {e}", exc_info=True
        )

    logger.debug(
        f"{model_type_name} モデルの登録が完了しました。登録済み合計: {registered_count}。"
        f"最終レジストリ状態: {list(registry.keys())}"
    )


# --- パブリックAPI ---


def register_annotators() -> dict[str, ModelClass]:
    """
    設定ファイルに基づいて、利用可能なすべてのアノテータクラスを登録します。

    Returns:
        dict[str, ModelClass]: 登録されたモデルクラスのレジストリ。
    """
    logger.info("すべてのアノテータを登録中...")
    _register_models(
        registry=_MODEL_CLASS_OBJ_REGISTRY,
        model_type_name="annotator",
        directory="model_class",
        base_module_path="image_annotator_lib.model_class",
        base_class=BaseAnnotator,
    )
    logger.info("アノテータの登録が完了しました。")
    return _MODEL_CLASS_OBJ_REGISTRY


def get_cls_obj_registry() -> dict[str, ModelClass]:
    """モデルクラスオブジェクトのレジストリを取得

    Note:
        このAPIは非推奨となる可能性があります。
        将来的にはより適切なインターフェースに置き換えられる可能性があります。

    Returns:
        dict[str, ModelClass]: 登録されたモデルクラスのレジストリ
    """
    return _MODEL_CLASS_OBJ_REGISTRY


def find_model_class_case_insensitive(model_name: str) -> tuple[str, ModelClass] | None:
    """大文字・小文字を区別しないモデルクラス検索

    Args:
        model_name: 検索するモデル名

    Returns:
        tuple[str, ModelClass] | None: (実際のキー名, モデルクラス) のタプル。見つからない場合はNone
    """
    # デバッグ情報を追加
    logger.debug(f"モデル検索: '{model_name}' を {len(_MODEL_CLASS_OBJ_REGISTRY)} 個のモデルから検索")
    logger.debug(f"利用可能なモデル: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())[:10]}...")

    # 最初に正確なマッチを試す
    if model_name in _MODEL_CLASS_OBJ_REGISTRY:
        logger.debug(f"正確なマッチ: '{model_name}'")
        return (model_name, _MODEL_CLASS_OBJ_REGISTRY[model_name])

    # 大文字・小文字を区別しない検索
    model_name_lower = model_name.lower()
    for key, value in _MODEL_CLASS_OBJ_REGISTRY.items():
        if key.lower() == model_name_lower:
            logger.debug(f"モデル名を正規化: '{model_name}' -> '{key}'")
            return (key, value)

    # 部分マッチを試す(ハイフンやスペースを無視)
    normalized_search = model_name.lower().replace("-", "").replace(" ", "")
    for key, value in _MODEL_CLASS_OBJ_REGISTRY.items():
        normalized_key = key.lower().replace("-", "").replace(" ", "")
        if normalized_search in normalized_key or normalized_key in normalized_search:
            logger.debug(f"部分マッチでモデル名を正規化: '{model_name}' -> '{key}'")
            return (key, value)

    logger.warning(
        f"モデル '{model_name}' が見つかりません。利用可能なモデル: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())[:5]}"
    )
    return None


# list_available_annotators のような統合されたリスト関数に変更 (あるいは既存を維持)
def list_available_annotators() -> list[str]:
    """利用可能なアノテータモデルの名前のリストを返します。"""
    return list(_MODEL_CLASS_OBJ_REGISTRY.keys())


_VALID_MODEL_TYPES: frozenset[str] = frozenset(("tagger", "scorer", "captioner", "vision"))


def _determine_model_type(
    model_name: str, model_class: ModelClass, model_config: dict[str, Any]
) -> ModelType:
    """モデルタイプを判定する。

    優先順位:
      1. 設定ファイルの `type` フィールド (annotator_config.toml で実際に使用)
      2. モデル名・クラス名からのキーワード推論
      3. デフォルト "vision"

    Args:
        model_name: モデル名
        model_class: モデルクラス
        model_config: モデル設定

    Returns:
        ModelType: "tagger" / "scorer" / "captioner" / "vision" のいずれか
    """
    # 設定の `type` を最優先で参照 (実際の config キーは "type")
    config_type = model_config.get("type")
    if isinstance(config_type, str) and config_type in _VALID_MODEL_TYPES:
        return cast(ModelType, config_type)

    # モデル名やクラス名から推定
    model_name_lower = model_name.lower()
    class_name_lower = model_class.__name__.lower()

    # スコア系モデルの判定
    if any(keyword in model_name_lower for keyword in ["aesthetic", "score", "rating", "quality"]):
        return "scorer"
    if any(keyword in class_name_lower for keyword in ["aesthetic", "score", "rating", "quality"]):
        return "scorer"

    # タガー系モデルの判定
    if any(keyword in model_name_lower for keyword in ["tagger", "tag", "wd", "deepdanbooru"]):
        return "tagger"
    if any(keyword in class_name_lower for keyword in ["tagger", "tag", "wd", "deepdanbooru"]):
        return "tagger"

    # キャプショナー系モデルの判定
    if any(keyword in model_name_lower for keyword in ["caption", "blip", "git"]):
        return "captioner"
    if any(keyword in class_name_lower for keyword in ["caption", "blip", "git"]):
        return "captioner"

    # 汎用 vision モデル (デフォルト)
    return "vision"


def _requires_api_key(model_class: ModelClass, model_config: dict[str, Any]) -> bool:
    """APIキーが必要かどうかを判定する

    Args:
        model_class: モデルクラス
        model_config: モデル設定

    Returns:
        bool: APIキーが必要かどうか
    """
    # 設定から直接取得できる場合
    if "requires_api_key" in model_config:
        return bool(model_config["requires_api_key"])

    # クラス名から判定
    class_name = model_class.__name__

    # PydanticAI WebAPI annotator は全てAPIキー必要
    if class_name == "PydanticAIWebAPIAnnotator":
        return True

    # 古いAPI系クラス名の判定
    if "api" in class_name.lower() or "webapi" in class_name.lower():
        return True

    # api_model_id が設定されている場合はAPI系
    if model_config.get("api_model_id"):
        return True

    # デフォルトはローカルモデル(APIキー不要)
    return False


def _build_annotator_info_for_registry_model(
    model_name: str, model_class: ModelClass, model_config: dict[str, Any]
) -> AnnotatorInfo:
    """レジストリ登録済みモデルから AnnotatorInfo を構築する。

    Args:
        model_name: モデル名 (レジストリキー)
        model_class: モデルクラス
        model_config: TOML 由来の設定辞書 (空辞書の場合もある)

    Returns:
        AnnotatorInfo: 型安全なメタデータ
    """
    from .utils import get_model_capabilities

    is_api = _requires_api_key(model_class, model_config)
    is_local = not is_api
    device = model_config.get("device") if is_local else None

    return AnnotatorInfo(
        name=model_name,
        model_type=_determine_model_type(model_name, model_class, model_config),
        capabilities=frozenset(get_model_capabilities(model_name)),
        is_local=is_local,
        is_api=is_api,
        device=device if isinstance(device, str) else None,
    )


def _build_annotator_info_for_direct_model(model_id: str) -> AnnotatorInfo:
    """PydanticAI 直接モデル (例: ``google/gemini-2.5-pro``) から AnnotatorInfo を構築する。

    これらのモデルはレジストリには登録されておらず、SimplifiedAgentFactory 経由で
    実行される。すべて WebAPI 呼び出しなので ``is_api=True``、device は持たない。

    capabilities は SimplifiedAgentWrapper.ADVERTISED_CAPABILITIES を参照する。
    申告値と実装が常に一致するよう、循環インポートを避けるため関数内で遅延 import する。

    Args:
        model_id: ``provider/model_name`` 形式のモデル ID

    Returns:
        AnnotatorInfo: 型安全なメタデータ。model_type は "vision" 固定 (汎用 VLM)。
    """
    from .simplified_agent_wrapper import SimplifiedAgentWrapper  # 循環 import 回避のため遅延

    return AnnotatorInfo(
        name=model_id,
        model_type="vision",
        capabilities=SimplifiedAgentWrapper.ADVERTISED_CAPABILITIES,
        is_local=False,
        is_api=True,
        device=None,
    )


def normalize_model_name(model_name: str) -> str | None:
    """モデル名を正規化(大文字・小文字を区別しない検索で実際のキー名を返す)

    Args:
        model_name: 正規化するモデル名

    Returns:
        str | None: 実際のレジストリキー名。見つからない場合はNone
    """
    result = find_model_class_case_insensitive(model_name)
    return result[0] if result else None


def _register_webapi_models_from_discovery() -> None:
    """available_api_models.toml を読み込み、WebAPI モデルをレジストリに直接登録する。

    annotator_config.toml への書き込みを行わず、廃止済みモデルを除外する。
    """
    logger.debug("available_api_models.toml から WebAPI モデルの直接登録を開始します...")
    try:
        api_models = load_available_api_models()
        if not api_models:
            logger.debug("利用可能な Web API モデル情報が見つかりません。登録をスキップします。")
            return

        logger.debug(f"{len(api_models)} 件の Web API モデル情報をロードしました。")

        available_classes = _gather_available_classes("model_class")
        pydantic_ai_class = available_classes.get("PydanticAIWebAPIAnnotator")
        if not pydantic_ai_class:
            logger.error(
                "PydanticAIWebAPIAnnotator クラスが見つかりません。WebAPI モデルは登録できません。"
            )
            return

        registered_count = 0
        for model_id, model_info in api_models.items():
            if not isinstance(model_info, dict):
                logger.warning(f"モデルID '{model_id}' の情報形式が不正です。スキップします。")
                continue

            if model_info.get("deprecated_on") is not None:
                continue

            model_name_short = model_info.get("model_name_short")
            provider = model_info.get("provider")

            if not model_name_short or not provider:
                logger.warning(
                    f"モデルID '{model_id}' に model_name_short または provider がありません。スキップします。"
                )
                continue

            if _try_register_model(
                _MODEL_CLASS_OBJ_REGISTRY, model_name_short, pydantic_ai_class, BaseAnnotator
            ):
                registered_count += 1
                metadata = {
                    "api_model_id": model_id,
                    "provider": provider,
                    "max_output_tokens": 1800,
                    "type": "webapi",
                    "class": "PydanticAIWebAPIAnnotator",
                }
                _WEBAPI_MODEL_METADATA[model_name_short] = metadata
                # PydanticAIWebAPIAnnotator が config_registry 経由で api_model_id を読むため
                # ファイルに書かずにメモリ内の merged config に登録する
                config_registry.set_system_value(model_name_short, "api_model_id", model_id)
                config_registry.set_system_value(model_name_short, "class", "PydanticAIWebAPIAnnotator")

        logger.info(f"WebAPI モデルの直接登録が完了しました。登録済み: {registered_count} 件。")

    except Exception as e:
        logger.error(f"WebAPI モデルの直接登録中に予期せぬエラーが発生しました: {e}", exc_info=True)


def _discover_and_update_api_models(skip_api_discovery: bool) -> None:
    """Web APIモデル情報の取得と設定ファイルの自動更新を行う。

    Args:
        skip_api_discovery: Trueの場合、API検出をスキップする。
    """
    ttl_expired = False
    if skip_api_discovery:
        logger.info(
            "環境変数 IMAGE_ANNOTATOR_SKIP_API_DISCOVERY=true のため、"
            "API モデル情報の取得をスキップします。"
        )
    elif not AVAILABLE_API_MODELS_CONFIG_PATH.exists():
        # 初回起動: 同期 fetch（後続の annotator_config 更新にデータが必要）
        logger.info(f"{AVAILABLE_API_MODELS_CONFIG_PATH} が見つかりません。APIから最新情報を取得します...")
        try:
            api_model_discovery._fetch_and_update_vision_models()
            logger.info(f"API からモデル情報を取得し、{AVAILABLE_API_MODELS_CONFIG_PATH} を更新しました。")
        except Exception as api_e:
            logger.error(f"API からのモデル情報取得中にエラーが発生しました: {api_e}", exc_info=True)
            logger.warning(
                "APIからのモデル情報取得に失敗したため、Web API モデルの自動設定は行われない可能性があります。"
            )
    elif api_model_discovery.should_refresh():
        # TTL 超過: _update_config_with_api_models() のファイル読み取り後に起動する（書き込み競合を避けるため）
        ttl_expired = True
    else:
        logger.debug("API モデル情報は TTL 内のため、既存キャッシュを使用します。")

    # available_api_models.toml からWebAPIモデルをレジストリに直接登録
    if AVAILABLE_API_MODELS_CONFIG_PATH.exists():
        _register_webapi_models_from_discovery()
    else:
        logger.warning(
            f"{AVAILABLE_API_MODELS_CONFIG_PATH} が存在しないため、Web API モデルの登録をスキップします。"
        )

    # ファイル読み取り完了後にバックグラウンド refresh を起動（read/write 競合を排除）
    if ttl_expired:
        api_model_discovery.trigger_background_refresh()
        logger.info(
            "TTL 超過のため API モデル情報の background refresh を起動しました。"
            "次回起動時から最新モデル一覧が反映されます。"
        )


def initialize_registry() -> None:
    """loggerとレジストリの初期化を明示的に行う関数。

    必ずlogger初期化後に呼び出すこと。
    Web API モデル情報の取得と設定ファイルの自動更新も行う。
    """
    global _REGISTRY_INITIALIZED

    from .utils import init_logger

    init_logger()

    if _REGISTRY_INITIALIZED:
        logger.debug(f"レジストリは既に初期化済みです。登録済みモデル数: {len(_MODEL_CLASS_OBJ_REGISTRY)}")
        return

    logger.debug("レジストリ初期化プロセスを開始します...")

    skip_api_discovery = os.getenv("IMAGE_ANNOTATOR_SKIP_API_DISCOVERY", "false").lower() == "true"

    try:
        _discover_and_update_api_models(skip_api_discovery)
    except Exception as e:
        logger.error(f"Web API モデル情報の処理中にエラーが発生しました: {e}", exc_info=True)

    logger.debug("アノテータ登録を開始します...")
    register_annotators()
    _REGISTRY_INITIALIZED = True
    logger.debug(f"レジストリの初期化が完了しました。登録済みアノテータ: {len(_MODEL_CLASS_OBJ_REGISTRY)}")
