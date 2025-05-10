import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import TypeVar

from . import api_model_discovery
from .base import BaseAnnotator
from .config import AVAILABLE_API_MODELS_CONFIG_PATH, config_registry, load_available_api_models
from .utils import logger

T = TypeVar("T", bound=BaseAnnotator)
ModelClass = type[BaseAnnotator]

_MODEL_CLASS_OBJ_REGISTRY: dict[str, ModelClass] = {}


# --- プライベートヘルパー関数 ---


def _list_module_files(directory: str) -> list[Path]:
    """指定されたディレクトリ内の全てのPythonモジュールファイル（サブディレクトリ含む、__init__.py除く）をリストアップ"""
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


def _recursive_subclasses(cls: type[T]) -> set[type[T]]:
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
                        available[subcls.__name__] = subcls
                elif hasattr(
                    obj, "predict"
                ):  # predictを持つがBaseAnnotatorサブクラスでない場合も登録 (元のロジック踏襲)
                    # このケースでは型が合わない可能性があるため注意が必要だが、元のロジックを維持
                    # 必要であれば、より厳密な型チェックや別の扱いを検討
                    available[name] = obj  # type: ignore

    logger.debug(f"{directory} から利用可能なクラス: {list(available.keys())}")
    return available


def _register_models(
    registry: dict[str, ModelClass],  # ModelClass を使用
    model_type_name: str,  # ログ用の "annotator" など
    directory: str,  # モデルクラスファイルを含むディレクトリ (例: "model_class")
    base_module_path: str,  # 基本Pythonインポートパス (例: "image_annotator_lib.model_class")
    base_class: type,  # モデルが継承すべき基底クラス (ABC互換性のために 'type' を使用)
    # config_filter は削除
) -> None:
    """設定に基づいてモデルを登録する汎用関数。"""
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

    registered_count = 0  # 変数を初期化

    try:
        # config_registry からロード済みの設定データを取得
        config = config_registry.get_all_config()
        logger.debug(f"[DEBUG _register_models] ロードされた設定 (config_registry.get_all_config()): {config}")

        if not config:
            logger.warning("モデル設定が空か、ロードに失敗しました。モデルは登録されません。")
            return
        logger.debug(f"モデル設定をロードしました: 合計 {len(config)} エントリ。")

        # 指定されたディレクトリから利用可能なクラスを収集
        available_classes = _gather_available_classes(directory)
        logger.debug(f"[DEBUG _register_models] 利用可能なクラス (_gather_available_classesの結果): {list(available_classes.keys())}")
        logger.debug(
            f"{len(available_classes)} 個の利用可能な {model_type_name} クラスが見つかりました: {list(available_classes.keys())}"
        )

        # 設定ファイルに基づいてモデルを登録
        for model_name, model_config in config.items():
            logger.debug(f"[DEBUG _register_models] 設定エントリを処理中: モデル名='{model_name}', 設定={model_config}")
            # config_filter によるフィルタリングを削除

            desired_class_name = model_config.get("class")
            if not desired_class_name:
                logger.warning(
                    f"設定でモデル '{model_name}' のクラス名が指定されていません。スキップします。"
                )
                continue
            logger.debug(f"[DEBUG _register_models] 期待されるクラス名: '{desired_class_name}'")

            # 収集された利用可能なクラスからクラスを検索
            model_cls = available_classes.get(desired_class_name)
            logger.debug(f"[DEBUG _register_models] 利用可能なクラスから '{desired_class_name}' を検索した結果: {model_cls}")

            if model_cls:
                # 期待されるbase_classのサブクラスであるか、または predict を持つかを確認 (元のロジック踏襲)
                if issubclass(model_cls, base_class) or hasattr(model_cls, "predict"):
                    if model_name in registry:
                        logger.warning(
                            f"モデル名 '{model_name}' は既に登録されています。クラス '{model_cls.__name__}' で上書きします。"
                        )
                    registry[model_name] = model_cls
                    registered_count += 1 # 登録成功時にカウント (修正)
                    logger.debug(f"[DEBUG _register_models] モデル '{model_name}' をクラス '{model_cls.__name__}' でレジストリに登録しました。現在のレジストリキー: {list(registry.keys())}")
                else:
                    logger.error(
                        f"モデル '{model_name}' のクラス '{desired_class_name}' が見つかりましたが、{base_class.__name__} を継承しておらず predict メソッドも持ちません。スキップします。"
                    )
            else:
                # 設定で指定されたクラスがスキャンされたディレクトリで見つかりません
                logger.warning(
                    f"{model_type_name} モデル '{model_name}' で指定されたクラス '{desired_class_name}' が、'{directory}' 内の利用可能なクラスの中に見つかりません。"
                    f"クラス名が正しいか、それを定義するファイルが '{base_module_path}' に存在するか確認してください。"
                )

    except Exception as e:
        logger.error(
            f"予期せぬエラーのため {model_type_name} モデルの登録に失敗しました: {e}", exc_info=True
        )

    logger.debug(
        f"{model_type_name} モデルの登録が完了しました。登録済み合計: {registered_count}。最終レジストリ状態: {list(registry.keys())}"
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


# list_available_annotators のような統合されたリスト関数に変更 (あるいは既存を維持)
def list_available_annotators() -> list[str]:
    """利用可能なアノテータモデルの名前のリストを返します。"""
    return list(_MODEL_CLASS_OBJ_REGISTRY.keys())


def _find_annotator_class_by_provider(provider: str, available_classes: dict[str, ModelClass]) -> str:
    """プロバイダー名に基づいてアノテータークラス名を検索する。

    - provider が google, openai, anthropic の場合、名前に provider が含まれるクラスを探す。
    - それ以外の場合、または一致するクラスが見つからない場合は OpenRouterApiAnnotator を返す。
    """
    provider_lower = provider.lower()
    specific_providers = {"google", "openai", "anthropic"}

    if provider_lower in specific_providers:
        # 特定プロバイダーの場合のみクラス名検索
        for class_name in available_classes:
            if provider_lower in class_name.lower():
                logger.debug(f"プロバイダー '{provider}' に一致するクラスが見つかりました: {class_name}")
                return class_name
        # 特定プロバイダーだが一致するクラスがなかった場合もフォールバック
        logger.warning(
            f"プロバイダー '{provider}' に一致するクラスが見つかりません。OpenRouterApiAnnotator を使用します。"
        )
        return "OpenRouterApiAnnotator"
    else:
        # 特定プロバイダー以外は OpenRouter を使用
        logger.debug(
            f"プロバイダー '{provider}' は特定プロバイダーではないため、OpenRouterApiAnnotator を使用します。"
        )
        return "OpenRouterApiAnnotator"


def _update_config_with_api_models() -> None:
    """available_api_models.toml を読み込み、annotator_config.toml にデフォルト設定を追加する。"""
    logger.debug("Web API モデルに基づいて annotator_config.toml の更新を開始します...")
    try:
        # 利用可能なAPIモデル情報をロード
        api_models = load_available_api_models()
        if not api_models:
            logger.debug("利用可能な Web API モデル情報が見つかりません。設定の更新をスキップします。")
            return

        logger.debug(f"{len(api_models)} 件の利用可能な Web API モデル情報をロードしました。")

        # 利用可能なアノテータークラスを収集
        available_classes = _gather_available_classes("model_class")
        if not available_classes:
            logger.warning(
                "利用可能なアノテータークラスが見つかりません。クラス名のマッピングができません。"
            )
            # 続行するが、クラス名は OpenRouterApiAnnotator にフォールバックされる

        # 各APIモデルについて設定を追加
        for model_id, model_info in api_models.items():
            if not isinstance(model_info, dict):
                logger.warning(
                    f"モデルID '{model_id}' の情報形式が不正です (辞書ではありません)。スキップします。"
                )
                continue

            model_name_short = model_info.get("model_name_short")
            provider = model_info.get("provider")

            if not model_name_short or not provider:
                logger.warning(
                    f"モデルID '{model_id}' の情報に model_name_short または provider がありません。スキップします。"
                )
                continue

            # プロバイダー名からクラス名を決定
            if ":" in model_id:  # モデルIDに `:` が含まれている場合は、`OpenRouterApiAnnotator` を使用する
                target_class_name = "OpenRouterApiAnnotator"
            else:
                target_class_name = _find_annotator_class_by_provider(provider, available_classes)

            # デフォルト設定を追加 (class と max_output_tokens)
            # add_default_setting はキーが存在しない場合のみ追加し、自動保存する
            config_registry.add_default_setting(model_name_short, "class", target_class_name)
            config_registry.add_default_setting(model_name_short, "max_output_tokens", 1800)

        logger.debug("annotator_config.toml の Web API モデル設定の更新が完了しました。")

    except Exception as e:
        logger.error(
            f"annotator_config.toml の自動更新中に予期せぬエラーが発生しました: {e}", exc_info=True
        )


def initialize_registry() -> None:
    """
    loggerとレジストリの初期化を明示的に行う関数。
    必ずlogger初期化後に呼び出すこと。
    Web API モデル情報の取得と設定ファイルの自動更新も行う。
    """
    from .utils import init_logger  # init_logger はここでインポート

    init_logger()
    logger.debug("レジストリ初期化プロセスを開始します...")

    # --- Web API モデル情報の取得と設定ファイルの自動更新 --- #
    try:
        if not AVAILABLE_API_MODELS_CONFIG_PATH.exists():
            logger.info(
                f"{AVAILABLE_API_MODELS_CONFIG_PATH} が見つかりません。APIから最新情報を取得します..."
            )
            try:
                # APIから取得してtomlファイルを生成/更新
                api_model_discovery._fetch_and_update_vision_models()
                logger.info(
                    f"API からモデル情報を取得し、{AVAILABLE_API_MODELS_CONFIG_PATH} を更新しました。"
                )
            except Exception as api_e:
                # API取得に失敗しても、処理は続行する（ログには残す）
                logger.error(f"API からのモデル情報取得中にエラーが発生しました: {api_e}", exc_info=True)
                logger.warning(
                    "APIからのモデル情報取得に失敗したため、Web API モデルの自動設定は行われない可能性があります。"
                )
        else:
            logger.debug(f"{AVAILABLE_API_MODELS_CONFIG_PATH} が存在します。既存のファイルを使用します。")

        # available_api_models.toml を読み込み、annotator_config.toml を更新
        _update_config_with_api_models()

    except Exception as e:
        # このステップ全体でエラーが発生しても初期化は続行する
        logger.error(f"Web API モデル情報の処理中にエラーが発生しました: {e}", exc_info=True)
    # --- ここまで --- #

    logger.debug("アノテータ登録を開始します...")
    register_annotators()
    logger.debug(f"レジストリの初期化が完了しました。登録済みアノテータ: {len(_MODEL_CLASS_OBJ_REGISTRY)}")
