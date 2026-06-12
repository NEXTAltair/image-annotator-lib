"""ローカル ML モデルの明示インストール API (LoRAIro Issue #754)。

推論時の暗黙 HuggingFace ダウンロードを、進捗コールバック付きの明示
ダウンロードとして切り出す。consumer (LoRAIro 等) はアノテーション実行前に
`is_model_installed()` で未インストールモデルを検出し、`install_model()` を
install ジョブとして実行できる。

設計メモ:
    - インストール済み判定は install marker ファイル (cache_dir/install_markers/)
      で行う。実ファイルのキャッシュ走査は from_pretrained 系の部分キャッシュと
      整合させるのが難しく、marker 方式が最も決定的で高速 (offline-safe)。
      本 API 導入前に推論経由で暗黙ダウンロード済みのモデルは「未インストール」
      と判定されるが、install_model() は HF cache を再利用するため再ダウンロード
      は発生せず短時間で完了する。
    - 進捗は huggingface_hub.snapshot_download の byte 集約進捗バー
      (tqdm_class フック) から取得する。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

from tqdm.auto import tqdm as _base_tqdm

from .core.config import config_registry
from .core.constants import DEFAULT_PATHS
from .core.utils import DEFAULT_TIMEOUT, logger
from .exceptions.errors import ModelInstallCancelledError, ModelNotFoundError

__all__ = [
    "ModelInstallProgress",
    "install_model",
    "is_model_installed",
]

ProgressCallback = Callable[["ModelInstallProgress"], None]

_INSTALL_MARKER_DIR_NAME = "install_markers"

# ONNX 系クラスが実際に参照するファイルのみを取得する allow_patterns
# (core/base/onnx.py の onnx_model_filename / onnx_metadata_* と同期)
_ONNX_ALLOW_PATTERNS: dict[str, list[str]] = {
    "WDTagger": ["model.onnx", "*.csv"],
    "Z3D_E621Tagger": ["model.onnx", "*.csv"],
    "CamieTagger": ["model_initial.onnx", "*.json"],
}

# transformers / CLIP 系の snapshot で除外する重複・不要 weight 形式
_TRANSFORMERS_STATIC_IGNORE_PATTERNS: list[str] = [
    "*.h5",
    "*.msgpack",
    "*.tflite",
    "*.ot",
    "*.onnx",
    "*.md",
    ".gitattributes",
]

# safetensors が存在する repo で重複ダウンロードを避ける追加 ignore
_DUPLICATE_WEIGHT_PATTERNS: list[str] = ["*.bin", "*.pth", "*.ckpt"]


@dataclass(frozen=True)
class ModelInstallProgress:
    """1 モデルのインストール進捗 (byte 単位)。

    Attributes:
        model_name: インストール対象のモデル名。
        downloaded_bytes: ダウンロード済みバイト数 (累積)。
        total_bytes: 総バイト数。メタデータ取得中は増加し得る。0 は未確定。
    """

    model_name: str
    downloaded_bytes: int
    total_bytes: int

    @property
    def percentage(self) -> int:
        """0-100 の進捗率を返す。総量未確定 (total_bytes<=0) は 0。"""
        if self.total_bytes <= 0:
            return 0
        return min(100, int(self.downloaded_bytes * 100 / self.total_bytes))


class _ByteProgressAggregator:
    """複数リソース / 複数スレッドからの byte 進捗を集約して callback へ流す。"""

    def __init__(
        self,
        model_name: str,
        progress_callback: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> None:
        self._model_name = model_name
        self._progress_callback = progress_callback
        self._cancel_event = cancel_event
        self._lock = threading.Lock()
        self._downloaded = 0
        self._total = 0

    def check_cancelled(self) -> None:
        """cancel_event がセットされていれば ModelInstallCancelledError を送出する。"""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise ModelInstallCancelledError(self._model_name)

    def add_total(self, n: int) -> None:
        """総バイト数を加算して進捗を通知する。"""
        if n <= 0:
            return
        with self._lock:
            self._total += n
            snapshot = self._snapshot_locked()
        self._emit(snapshot)

    def add_downloaded(self, n: int) -> None:
        """ダウンロード済みバイト数を加算して進捗を通知する。"""
        if n <= 0:
            return
        with self._lock:
            self._downloaded += n
            snapshot = self._snapshot_locked()
        self._emit(snapshot)

    def _snapshot_locked(self) -> ModelInstallProgress:
        return ModelInstallProgress(
            model_name=self._model_name,
            downloaded_bytes=self._downloaded,
            total_bytes=self._total,
        )

    def _emit(self, snapshot: ModelInstallProgress) -> None:
        if self._progress_callback is not None:
            self._progress_callback(snapshot)


class _SnapshotProgressTqdm(_base_tqdm):
    """snapshot_download の byte 集約バーを aggregator へ橋渡しする tqdm。

    huggingface_hub は `tqdm_class` で渡したクラスを (1) byte 集約バー
    (unit="B")、(2) ファイル数バー (thread_map) の両方に使う。byte バーのみを
    追跡し、コンソール表示は disable する。

    Note:
        意図的に huggingface_hub.utils.tqdm ではなく素の tqdm を継承する。
        hf サブクラスを継承すると `_create_progress_bar` が disable/name を
        注入してくるため、挙動を自前で制御できる素の tqdm 経路を使う。
        aggregator は `_build_snapshot_tqdm_class()` がサブクラスの
        ClassVar として束縛する (hf 側がクラスを直接インスタンス化するため
        コンストラクタ経由で渡せない)。
    """

    _aggregator: ClassVar[_ByteProgressAggregator]

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._tracks_bytes = kwargs.get("unit") == "B"
        self._reported_total = 0
        kwargs["disable"] = True  # 進捗は callback 経由で通知、コンソール表示は不要
        if self._tracks_bytes:
            initial = kwargs.get("initial")
            if isinstance(initial, int) and initial > 0:
                self._aggregator.add_downloaded(initial)
        super().__init__(*args, **kwargs)
        if self._tracks_bytes:
            self._sync_total()

    def _sync_total(self) -> None:
        """外部から直接書き換えられる total 属性の増分を aggregator へ反映する。"""
        current_total = int(self.total or 0)
        delta = current_total - self._reported_total
        if delta > 0:
            self._reported_total = current_total
            self._aggregator.add_total(delta)

    def refresh(self, *args: object, **kwargs: object) -> bool:
        # hf 側はファイルメタデータ取得ごとに total を直接加算して refresh() を呼ぶ
        self._aggregator.check_cancelled()
        if self._tracks_bytes:
            self._sync_total()
        return bool(super().refresh(*args, **kwargs))

    def update(self, n: int | float | None = 1) -> bool | None:
        self._aggregator.check_cancelled()
        if self._tracks_bytes and n is not None and n > 0:
            self._aggregator.add_downloaded(int(n))
        return super().update(n)


def _build_snapshot_tqdm_class(aggregator: _ByteProgressAggregator) -> type[_SnapshotProgressTqdm]:
    """aggregator を束縛した _SnapshotProgressTqdm サブクラスを返す。"""
    return type("_BoundSnapshotProgressTqdm", (_SnapshotProgressTqdm,), {"_aggregator": aggregator})


def _marker_path(model_name: str) -> Path:
    """モデルの install marker ファイルパスを返す。"""
    # config セクション名にパス区切りは現れないが、防御的に置換する
    safe_name = model_name.replace("/", "__").replace("\\", "__")
    return Path(DEFAULT_PATHS["cache_dir"]) / _INSTALL_MARKER_DIR_NAME / f"{safe_name}.json"


def _model_config(model_name: str) -> dict[str, object] | None:
    """config registry からモデル設定を返す。未登録なら None。"""
    all_config = config_registry.get_all_config()
    config = all_config.get(model_name)
    if isinstance(config, dict):
        return config
    return None


def is_model_installed(model_name: str) -> bool:
    """モデルがインストール済み (明示ダウンロード完了済み) かを判定する。

    install marker ファイルで判定する高速・offline-safe な操作。marker に
    記録された model_path が現在の config と異なる場合 (config の付け替え) は
    stale とみなし False を返す。config registry に存在しないモデル
    (WebAPI モデル等、ダウンロード不要) は常に True を返す。

    Args:
        model_name: 判定対象のモデル名 (annotator_config.toml のセクション名)。

    Returns:
        インストール済み (またはインストール不要) なら True。
    """
    config = _model_config(model_name)
    if config is None or not config.get("model_path"):
        return True
    marker = _marker_path(model_name)
    try:
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"install marker の読み込み失敗 ({model_name}): {e} — 未インストール扱い")
        return False
    # Codex P2 (PR #151): config の model_path 付け替えで stale になった marker を拒否する
    return marker_data.get("model_path") == str(config.get("model_path"))


def install_model(
    model_name: str,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """モデルを明示的にダウンロード (インストール) する。

    モデルのクラス種別に応じて必要なリソース (HF repo / URL ファイル) を
    ダウンロードし、完了時に install marker を書き込む。HF cache 済みファイルは
    再ダウンロードされないため、暗黙ダウンロード済みモデルへの実行は短時間で
    完了する。

    Args:
        model_name: インストール対象のモデル名。
        progress_callback: byte 進捗の通知先 (任意)。ダウンロードスレッドから
            呼ばれるためスレッドセーフであること。
        cancel_event: キャンセル要求イベント (任意)。セットされると進捗更新の
            タイミングで ModelInstallCancelledError を送出して中断する。

    Raises:
        ModelNotFoundError: config registry に存在しないモデル名の場合。
        ModelInstallCancelledError: cancel_event によって中断された場合。
    """
    config = _model_config(model_name)
    if config is None:
        raise ModelNotFoundError(model_name)

    model_path = str(config.get("model_path") or "")
    if not model_path:
        logger.warning(f"モデル '{model_name}' に model_path がないため install をスキップ")
        return

    aggregator = _ByteProgressAggregator(model_name, progress_callback, cancel_event)
    model_class = str(config.get("class") or "")

    logger.info(f"モデルインストール開始: {model_name} (class={model_class})")
    parsed = urlparse(model_path)
    if parsed.scheme in ("http", "https"):
        _install_url_resource(model_path, aggregator)
    elif Path(model_path).exists():
        # Codex P2 (PR #151): ローカルファイル/ディレクトリ指定はダウンロード不要。
        # 既存 loader (get_file_path / from_pretrained) がそのまま解決できる。
        logger.debug(f"model_path はローカルパスのため DL をスキップ: {model_path}")
    else:
        _install_hf_repo(model_path, aggregator, _hf_patterns_for_class(config, model_class))

    base_model = str(config.get("base_model") or "")
    if base_model:
        # CLIP 系: 分類 head (URL) に加えて base CLIP repo が必要
        _install_hf_repo(base_model, aggregator, _HfPatterns(transformers_weights=True))

    _write_marker(model_name, model_path)
    logger.info(f"モデルインストール完了: {model_name}")


@dataclass(frozen=True)
class _HfPatterns:
    """HF repo ダウンロードのファイル選択パターン。"""

    allow_patterns: list[str] | None = None
    transformers_weights: bool = False


def _hf_patterns_for_class(config: dict[str, object], model_class: str) -> _HfPatterns:
    """モデルクラスに応じた HF ダウンロードパターンを返す。"""
    if model_class in _ONNX_ALLOW_PATTERNS:
        return _HfPatterns(allow_patterns=list(_ONNX_ALLOW_PATTERNS[model_class]))
    if model_class == "AnimeRatingAnnotator":
        variant = str(config.get("model_variant") or "")
        if variant:
            return _HfPatterns(allow_patterns=[f"{variant}/model.onnx", f"{variant}/meta.json"])
    # transformers / pipeline 系: from_pretrained 相当の snapshot (重複 weight 除外)
    return _HfPatterns(transformers_weights=True)


def _transformers_ignore_patterns(repo_id: str) -> list[str]:
    """transformers 系 repo の ignore_patterns を返す。

    safetensors が存在する repo では .bin/.pth/.ckpt を除外して重複ダウンロードを
    避ける (from_pretrained は safetensors を優先するため)。repo 一覧の取得に
    失敗した場合は静的パターンのみ返す。
    """
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    patterns = list(_TRANSFORMERS_STATIC_IGNORE_PATTERNS)
    try:
        repo_files = huggingface_hub.list_repo_files(repo_id)
    except (HfHubHTTPError, OSError) as e:
        logger.debug(f"repo ファイル一覧の取得失敗 ({repo_id}): {e} — 静的 ignore のみ適用")
        return patterns
    if any(name.endswith(".safetensors") for name in repo_files):
        patterns.extend(_DUPLICATE_WEIGHT_PATTERNS)
    return patterns


def _install_hf_repo(repo_id: str, aggregator: _ByteProgressAggregator, patterns: _HfPatterns) -> None:
    """HF repo を進捗付きで snapshot ダウンロードする。"""
    import huggingface_hub

    # Codex P2 (PR #151): ignore patterns 計算 (list_repo_files の network call) の
    # 前にキャンセルを確認し、キャンセル済みの場合に network へ出ないようにする
    aggregator.check_cancelled()

    ignore_patterns: list[str] | None = None
    if patterns.allow_patterns is None and patterns.transformers_weights:
        ignore_patterns = _transformers_ignore_patterns(repo_id)

    aggregator.check_cancelled()
    huggingface_hub.snapshot_download(
        repo_id,
        allow_patterns=patterns.allow_patterns,
        ignore_patterns=ignore_patterns,
        tqdm_class=_build_snapshot_tqdm_class(aggregator),
    )


def _install_url_resource(url: str, aggregator: _ByteProgressAggregator) -> None:
    """URL リソースを進捗・キャンセル対応でダウンロードする。

    `core/utils.py` の URL キャッシュ規約 (_get_cache_path) と同じ配置に保存する
    ため、インストール後の推論はキャッシュをそのまま再利用する。zip は推論時
    (`load_file`) に解凍されるためここでは展開しない。
    """
    import requests

    from .core import utils

    cache_dir = Path(DEFAULT_PATHS["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    is_cached, local_path = utils._is_cached(url, cache_dir)
    if is_cached:
        logger.debug(f"URL リソースはキャッシュ済み: {local_path}")
        return

    aggregator.check_cancelled()
    response = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    aggregator.add_total(total_size)

    # キャンセル時に壊れたキャッシュを残さないよう一時ファイルに書いて rename する
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                aggregator.check_cancelled()
                if chunk:
                    f.write(chunk)
                    aggregator.add_downloaded(len(chunk))
        tmp_path.replace(local_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_marker(model_name: str, model_path: str) -> None:
    """install 完了 marker を書き込む。"""
    marker = _marker_path(model_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_path": model_path,
                "installed_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
