import datetime
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

from .base import BaseAnnotator
from .config import config_registry
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
            # 古いプロバイダー固有のクラスは除外
            # ADR 0023 Phase 1 (Issue #35): WebApiAnnotator 1 種に統一されたため、
            # PydanticAIWebAPIAnnotator (旧統一クラス) と個別プロバイダー固有クラスは
            # 全て obsolete。user TOML に古い class 名が残っている場合の防衛として残置。
            obsolete_classes = [
                "PydanticAIWebAPIAnnotator",
                "AnthropicApiAnnotator",
                "GoogleApiAnnotator",
                "OpenAIApiChatAnnotator",
                "OpenAIApiResponseAnnotator",
            ]
            if name in obsolete_classes:
                logger.debug(
                    f"古いクラス '{name}' をスキップします (ADR 0023 Phase 1 で WebApiAnnotator に統合)"
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

    ADR 0023 Phase 1 (Issue #35) で `WebApiAnnotator` 1 種に統一。
    `PydanticAIWebAPIAnnotator` および個別プロバイダークラス
    (`OpenAIApiAnnotator` / `OpenAIApiChatAnnotator` /
    `OpenAIApiResponseAnnotator` 等) は obsolete として扱う。

    Args:
        class_name: チェック対象のクラス名。

    Returns:
        廃止クラスの場合True。
    """
    if class_name == "WebApiAnnotator":
        return False
    # PydanticAIWebAPIAnnotator は ADR 0023 Phase 1 で WebApiAnnotator に統合されたため obsolete
    if class_name == "PydanticAIWebAPIAnnotator":
        return True
    # *ApiAnnotator, *ApiChatAnnotator, *ApiResponseAnnotator を検出
    obsolete_suffixes = ("ApiAnnotator", "ApiChatAnnotator", "ApiResponseAnnotator")
    return any(class_name.endswith(suffix) for suffix in obsolete_suffixes)


def _resolve_model_class(
    desired_class_name: str,
    model_name: str,
    available_classes: dict[str, type],
    model_type_name: str,
) -> type | None:
    """設定エントリからモデルクラスを解決する (ローカル ML モデル専用)。

    ADR 0023 Phase 1: WebAPI 用 user TOML override は **廃止** (LiteLLM 未登録モデルは
    利用不可)。WebAPI モデルの registry 登録は `_register_webapi_models_from_discovery()`
    が LiteLLM 同梱 DB から SSoT として行うため、本関数は **ローカル ML モデルの class
    解決のみ** を担当する。

    user TOML が `class = "WebApiAnnotator"` を指定した場合は、ADR 0023 の決定に従い
    warning + skip する。`_register_webapi_models_from_discovery()` 経由では本関数は
    呼ばれない (`WebApiAnnotator` を直接 `_try_register_model()` に渡すため)。

    Args:
        desired_class_name: 設定で指定されたクラス名。
        model_name: ログ用のモデル名。
        available_classes: スキャン済みの利用可能なクラス辞書。
        model_type_name: ログ用のモデルタイプ名。

    Returns:
        解決されたモデルクラス、またはスキップすべき場合はNone。
    """
    # ADR 0023 Phase 1 (Codex P1, PR #40): user TOML 経由の WebAPI モデル定義は禁止。
    # WebApiAnnotator の registry 登録は LiteLLM 同梱 DB 由来の
    # `_register_webapi_models_from_discovery()` が排他的に行うため、user TOML 側で
    # `class = "WebApiAnnotator"` を指定しても registry には載せない (broken path 防止)。
    if desired_class_name == "WebApiAnnotator":
        logger.warning(
            f"モデル '{model_name}' で `class = 'WebApiAnnotator'` が user TOML から指定されています。"
            f"ADR 0023 Phase 1 以降、WebAPI モデル定義は LiteLLM 同梱 DB が SSoT で、"
            f"user TOML 経由の WebAPI モデル定義はサポート対象外です。"
            f"LiteLLM DB に登録されたモデル ID (例: 'openai/gpt-4o') を直接 model_name に "
            f"指定してください。本エントリはスキップします。"
        )
        return None

    # 古いプロバイダー固有クラス・PydanticAIWebAPIAnnotator は警告してスキップ
    if _is_obsolete_annotator_class(desired_class_name):
        logger.warning(
            f"モデル '{model_name}' で旧クラス '{desired_class_name}' が指定されています。"
            f"ADR 0023 Phase 1 以降は WebAPI モデル定義に user TOML を使用しません "
            f"(LiteLLM DB 由来の自動登録のみサポート)。スキップします。"
        )
        return None

    # ローカル ML モデルの class 解決
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
            logger.debug(f"モデル名 '{model_name}' は既に登録されています(同一クラス)。スキップします。")
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

        # ADR 0023 Phase 1 (Codex P1, PR #40): WebAPI モデルは _register_webapi_models_from_discovery()
        # が LiteLLM 同梱 DB から SSoT として登録するため、`_register_models()` 経由で
        # user TOML の `class = "WebApiAnnotator"` を解決することはない (本ループは
        # ローカル ML モデルの解決のみを担当する)。

        for model_name, model_config in config.items():
            desired_class_name = model_config.get("class")
            if not desired_class_name:
                logger.warning(
                    f"設定でモデル '{model_name}' のクラス名が指定されていません。スキップします。"
                )
                continue

            model_cls = _resolve_model_class(
                desired_class_name, model_name, available_classes, model_type_name
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


def get_webapi_metadata(model_name: str) -> dict[str, Any] | None:
    """登録済み WebAPI モデルのメタデータを返す。未登録なら ``None``。

    `_register_webapi_models_from_discovery` が `available_api_models.toml` から
    構築する **WebAPI モデルメタデータの単一情報源 (SSoT)** を提供する getter。
    呼び出し側はモジュール内の private 辞書 ``_WEBAPI_MODEL_METADATA`` を直接 import せず、
    本関数を経由してアクセスすること。

    Args:
        model_name: ``model_name_short`` (例: ``"GPT-4o"``)。

    Returns:
        メタデータ辞書 (``litellm_model_id`` / ``provider`` / ``max_output_tokens`` /
        ``supports_vision`` / ``supports_response_schema`` / ``type`` / ``class`` などを含む)。
        未登録なら ``None``。
    """
    return _WEBAPI_MODEL_METADATA.get(model_name)


_VALID_MODEL_TYPES: frozenset[str] = frozenset(("tagger", "scorer", "captioner", "vision", "rating"))


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
        ModelType: "tagger" / "scorer" / "captioner" / "vision" / "rating" のいずれか
    """
    # 設定の `type` を最優先で参照 (実際の config キーは "type")
    config_type = model_config.get("type")
    if isinstance(config_type, str) and config_type in _VALID_MODEL_TYPES:
        return cast(ModelType, config_type)

    # モデル名やクラス名から推定
    model_name_lower = model_name.lower()
    class_name_lower = model_class.__name__.lower()

    # レーティング専用モデルの判定
    if any(keyword in model_name_lower for keyword in ["anime_rating", "moderation"]):
        return "rating"
    if any(keyword in class_name_lower for keyword in ["ratingannotator", "moderation"]):
        return "rating"

    # スコア系モデルの判定
    if any(keyword in model_name_lower for keyword in ["aesthetic", "score", "quality"]):
        return "scorer"
    if any(keyword in class_name_lower for keyword in ["aesthetic", "score", "quality"]):
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
    """APIキーが必要かどうかを判定する。

    ADR 0023 Phase 1 (Issue #35, PR #40): WebAPI 判定は `WebApiAnnotator` サブクラス
    判定のみに統一。旧 `api_model_id` フォールバック (TOML 由来 metadata 後方互換) は
    廃止された (ADR 0023 Phase 1: WebAPI 用 user TOML override 廃止に伴い不要)。

    Args:
        model_class: モデルクラス
        model_config: モデル設定

    Returns:
        bool: APIキーが必要かどうか
    """
    # 設定から直接取得できる場合
    if "requires_api_key" in model_config:
        return bool(model_config["requires_api_key"])

    # WebApiAnnotator (またはサブクラス) は API key 必須
    from .webapi_annotator import WebApiAnnotator

    if issubclass(model_class, WebApiAnnotator):
        return True

    # デフォルトはローカルモデル(APIキー不要)
    return False


def _resolve_registry_capabilities(model_name: str, is_api: bool) -> frozenset[TaskCapability]:
    """レジストリモデルの capabilities を解決する。

    設定ファイルに明示的な capabilities がある場合はそれを使用する。
    WebAPI モデル (is_api=True) で capabilities が未設定の場合は、
    `WebApiAnnotator.ADVERTISED_CAPABILITIES` (tags/captions/scores の 3 種) を申告する。

    Args:
        model_name: モデル名
        is_api: WebAPI モデルか

    Returns:
        frozenset[TaskCapability]
    """
    from .utils import get_model_capabilities

    caps = get_model_capabilities(model_name)
    if caps:
        return frozenset(caps)

    if is_api:
        # ADR 0023 Phase 1: WebApiAnnotator は AnnotationSchema (tags/captions/score) を返す
        from .webapi_annotator import WebApiAnnotator  # 循環 import 回避のため遅延

        return WebApiAnnotator.ADVERTISED_CAPABILITIES

    return frozenset()


def _infer_provider_from_model_id(model_id: str) -> str | None:
    """PydanticAI 直接モデルの ``provider`` を model_id から推論する。

    "provider/model_name" 形式 (例: "google/gemini-2.5-pro") は slash 前を返す。
    slash のない model_id は PydanticAI の infer_provider_class でフォールバック。

    戻り値は **常に lowercase** で正規化する (PR #27 Codex P2 反映): registry-backed
    WebAPI モデル / direct モデル / user TOML override の各経路で provider 名が
    case-sensitive に一貫することを保証する。

    Args:
        model_id: PydanticAI 直接モデルの ID。

    Returns:
        推論された provider 名 (小文字)。判定できない場合は None。
    """
    if "/" in model_id:
        return model_id.split("/", 1)[0].lower()
    try:
        from pydantic_ai.providers import infer_provider_class

        cls = infer_provider_class(model_id)
        cls_name = (type(cls).__name__ if not isinstance(cls, type) else cls.__name__).lower()
        for known in ("openai", "anthropic", "google"):
            if known in cls_name:
                return known
    except Exception:
        pass
    return None


def _build_annotator_info_for_registry_model(
    model_name: str, model_class: ModelClass, model_config: dict[str, Any]
) -> AnnotatorInfo:
    """レジストリ登録済みモデルから AnnotatorInfo を構築する。

    Args:
        model_name: モデル名 (レジストリキー)
        model_class: モデルクラス
        model_config: model_config の出所 (排他分岐):
            - WebAPI モデル: `_WEBAPI_MODEL_METADATA` (SSoT) + user TOML override
            - ローカル ML モデル: `config_registry` (user TOML)

    Returns:
        AnnotatorInfo: 型安全なメタデータ
    """
    is_api = _requires_api_key(model_class, model_config)
    is_local = not is_api
    device = model_config.get("device") if is_local else None

    # Phase 2 (Issue #19/#26): 詳細メタデータを model_config から取得。
    # `_safe_float` / `_safe_int` / `_parse_discontinued_at` は malformed 値で
    # warning + None フォールバックする (Codex P2 #1, #2, #5 の根本対応)。
    # `provider` はローカルモデルなら "local" にフォールバック (ADR 0005)。
    # provider は lowercase 正規化 (PR #27 Codex P2 反映): user TOML/discovery/直接モデルで
    # case が混在しても AnnotatorInfo.provider は常に小文字で一貫する。
    raw_provider = model_config.get("provider")
    # ローカルMLモデルは provider を "local" に固定 (Codex P2 r3193126501):
    # user TOML に誤って provider キーが含まれても is_local=True との矛盾を防ぐ。
    provider: str | None = (
        "local" if is_local else (str(raw_provider).lower() if raw_provider is not None else None)
    )

    # ADR 0023 Phase 2 (Issue #41): metadata の `litellm_model_id` SSoT を
    # `AnnotatorInfo.litellm_model_id` field に直接公開する。LoRAIro 側は ADR 0023
    # line 73 に従い `api_model_id` 互換シムを廃止し、`litellm_model_id` を読む。
    # ローカル ML モデルは外部 ID を持たないため None。
    raw_litellm_id = model_config.get("litellm_model_id") if is_api else None
    litellm_model_id: str | None = str(raw_litellm_id) if raw_litellm_id is not None else None

    return AnnotatorInfo(
        name=model_name,
        model_type=_determine_model_type(model_name, model_class, model_config),
        capabilities=_resolve_registry_capabilities(model_name, is_api),
        is_local=is_local,
        is_api=is_api,
        device=device if isinstance(device, str) else None,
        provider=provider,
        litellm_model_id=litellm_model_id,
        estimated_size_gb=_safe_float(
            model_config.get("estimated_size_gb"), model_name, "estimated_size_gb"
        ),
        discontinued_at=_parse_discontinued_at(model_config.get("discontinued_at"), model_name),
        max_output_tokens=_safe_int(model_config.get("max_output_tokens"), model_name, "max_output_tokens"),
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


def _safe_float(value: Any, model_name: str, field: str) -> float | None:
    """optional な数値メタデータを float に安全に変換する。

    変換できない値は **モデル登録を失敗させず**、warning ログ + None フォールバック
    する (ADR 0005: optional metadata の不正値で listing 全体が壊れない設計)。
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"モデル '{model_name}' の {field} を float に変換できません: {value!r}")
        return None


def _safe_int(value: Any, model_name: str, field: str) -> int | None:
    """optional な数値メタデータを int に安全に変換する。

    `_safe_float` と同方針。変換失敗時は warning + None フォールバック。
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(f"モデル '{model_name}' の {field} を int に変換できません: {value!r}")
        return None


def _parse_discontinued_at(value: Any, model_name: str) -> datetime.datetime | None:
    """``discontinued_at`` を ``datetime.datetime | None`` に正規化する。

    受け付ける型:
        - ``datetime.datetime``: そのまま返す
        - ``datetime.date`` (TOML local-date ``2025-12-31`` 等): UTC 00:00:00 の datetime に変換
        - ``str``: ISO 8601 として parse (``"Z"`` suffix は UTC として扱う)

    Note:
        `datetime.datetime` は `datetime.date` のサブクラスなので、isinstance チェックは
        必ず datetime → date の順に行う必要がある。
    """
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        # TOML local-date (時刻なし) は datetime.date として渡される。UTC 00:00:00 で datetime 化。
        return datetime.datetime.combine(value, datetime.time.min, tzinfo=datetime.UTC)
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning(f"モデル '{model_name}' の discontinued_at を datetime にパース失敗: {value!r}")
            return None
    logger.warning(
        f"モデル '{model_name}' の discontinued_at が予期しない型 ({type(value).__name__}): {value!r}"
    )
    return None


def _register_webapi_models_from_discovery() -> None:
    """LiteLLM 同梱 DB から WebAPI モデルをレジストリに直接登録する (ADR 0023 Phase 1)。

    旧 `available_api_models.toml` 経由の登録は廃止。`discover_available_vision_models()`
    の `metadata` (LiteLLM `get_model_info()` 由来) を直接使用する。

    Issue #35: 登録対象クラスは `WebApiAnnotator` (`core/webapi_annotator.py`) を
    直接 import して使う。旧 `PydanticAIWebAPIAnnotator` 経路は廃止。
    """
    logger.debug("LiteLLM 同梱 DB から WebAPI モデルの直接登録を開始します...")
    try:
        from .api_model_discovery import discover_available_vision_models
        from .webapi_annotator import WebApiAnnotator

        result = discover_available_vision_models()
        api_models: dict[str, dict[str, Any]] = result.get("metadata", {})
        if not api_models:
            logger.debug("利用可能な Web API モデル情報が見つかりません。登録をスキップします。")
            return

        logger.debug(f"{len(api_models)} 件の Web API モデル情報を取得しました。")

        registered_count = 0
        for model_id, model_info in api_models.items():
            if not isinstance(model_info, dict):
                logger.warning(f"モデルID '{model_id}' の情報形式が不正です。スキップします。")
                continue

            mode = model_info.get("mode", "chat")
            model_name_short = model_info.get("model_name_short")
            provider = model_info.get("provider")

            if not model_name_short or not provider:
                logger.warning(
                    f"モデルID '{model_id}' に model_name_short または provider がありません。スキップします。"
                )
                continue

            if _try_register_model(
                _MODEL_CLASS_OBJ_REGISTRY, model_name_short, WebApiAnnotator, BaseAnnotator
            ):
                registered_count += 1
                # `_WEBAPI_MODEL_METADATA` は WebAPI モデルメタデータの単一情報源 (SSoT)。
                # ADR 0023 Phase 1 (Issue #35, PR #40): 外部 API ID は `litellm_model_id`
                # を SSoT とする。旧 `api_model_id` キー重複は廃止 (ADR 0023 line 73:
                # 「互換シムを残さない」)。
                metadata = {
                    "litellm_model_id": model_id,
                    "model_name_on_provider": model_id,
                    "provider": str(provider).lower(),
                    "mode": mode,
                    "max_input_tokens": _safe_int(
                        model_info.get("max_input_tokens"), model_id, "max_input_tokens"
                    ),
                    "max_output_tokens": _safe_int(
                        model_info.get("max_output_tokens"), model_id, "max_output_tokens"
                    )
                    or 1800,
                    "max_tokens": _safe_int(model_info.get("max_tokens"), model_id, "max_tokens"),
                    "supports_vision": True,
                    # ADR 0023 Phase 1 (Issue #45): structured output は PydanticAI default
                    # Tool Output で得るため、`supports_response_schema` キーは metadata から
                    # 削除した。`supports_function_calling` を WebAPI モデル登録の主条件に統一。
                    "supports_function_calling": bool(model_info.get("supports_function_calling")),
                    "supports_tool_choice": bool(model_info.get("supports_tool_choice")),
                    "supports_parallel_function_calling": bool(
                        model_info.get("supports_parallel_function_calling")
                    ),
                    "input_cost_per_token": _safe_float(
                        model_info.get("input_cost_per_token"), model_id, "input_cost_per_token"
                    ),
                    "output_cost_per_token": _safe_float(
                        model_info.get("output_cost_per_token"), model_id, "output_cost_per_token"
                    ),
                    "estimated_size_gb": _safe_float(
                        model_info.get("estimated_size_gb"), model_id, "estimated_size_gb"
                    ),
                    "discontinued_at": None,
                    # Issue #82: discovery 由来の WebAPI モデルは Vision LLM であり、
                    # prompt 指示で tags/captions/scores に加え rating も出力できる。
                    # `capabilities` を明示しないと `get_model_capabilities` の fallback で
                    # RATINGS が欠落し、rating prompt / 正規化経路が production で到達不能になる。
                    "capabilities": ["tags", "captions", "scores", "ratings"],
                    "type": "webapi",
                    "class": "WebApiAnnotator",
                }
                _WEBAPI_MODEL_METADATA[model_name_short] = metadata

        logger.info(f"WebAPI モデルの直接登録が完了しました。登録済み: {registered_count} 件。")

    except Exception as e:
        logger.error(f"WebAPI モデルの直接登録中に予期せぬエラーが発生しました: {e}", exc_info=True)


def _discover_and_update_api_models() -> None:
    """LiteLLM 同梱 DB から WebAPI モデル情報を取得し、registry に登録する (ADR 0023 Phase 1)。

    旧 TOML cache / TTL refresh / OpenRouter fallback / background refresh はすべて廃止。
    LiteLLM 同梱 DB は network 通信を必要としないため、Phase 0 にあった
    `IMAGE_ANNOTATOR_SKIP_API_DISCOVERY` フラグも廃止された。テストで WebAPI モデル登録を
    抑制したい場合は `_register_webapi_models_from_discovery` を pytest fixture で
    monkeypatch すること。
    """
    _register_webapi_models_from_discovery()


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

    try:
        _discover_and_update_api_models()
    except Exception as e:
        logger.error(f"Web API モデル情報の処理中にエラーが発生しました: {e}", exc_info=True)

    logger.debug("アノテータ登録を開始します...")
    register_annotators()
    _REGISTRY_INITIALIZED = True
    logger.debug(f"レジストリの初期化が完了しました。登録済みアノテータ: {len(_MODEL_CLASS_OBJ_REGISTRY)}")
