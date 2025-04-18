import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import huggingface_hub
import imagehash
import requests
from loguru import logger
from PIL import Image
from tqdm import tqdm

# confing モジュールで定数を定義すると循環インポートになるのを回避
from .constants import DEFAULT_PATHS

DEFAULT_TIMEOUT = 30
WD_MODEL_FILENAME = "model.onnx"
WD_LABEL_FILENAME = "selected_tags.csv"


# ログフォーマット
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}"

# loguru のデフォルトハンドラを削除 (明示的に設定するため)
logger.remove()

# コンソールシンク (stderr)
logger.add(
    sys.stderr,
    level="INFO",  # デフォルトレベル (必要に応じて変更)
    format=LOG_FORMAT,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# ファイルシンク (DEFAULT_PATHS["log_file"] を使用)
try:
    log_file_path = Path(DEFAULT_PATHS["log_file"])
    log_file_path.parent.mkdir(parents=True, exist_ok=True)  # フォルダ作成
    logger.add(
        log_file_path,
        level="DEBUG",
        format=LOG_FORMAT,
        rotation="25 MB",
        retention=5,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )
    logger.info(f"Logging to file: {log_file_path}")
except Exception as e:
    # ファイルログ設定失敗時はエラーログを出力して続行 (コンソールには出力される)
    logger.error(f"Failed to configure file logging to '{DEFAULT_PATHS['log_file']}': {e}")
    logger.error("File logging disabled.")


def calculate_phash(image: Image.Image) -> str:
    """画像から pHash を計算します。

    Args:
        image: pHash を計算する PIL Image オブジェクト。

    Returns:
        str: pHash 文字列
    """
    rgb_image = image.convert("RGB")
    hash_val = imagehash.phash(rgb_image)
    return str(hash_val)


def _get_cache_path(url: str, cache_dir: Path) -> Path:
    """URLからキャッシュファイルパスを生成する"""
    filename = Path(urlparse(url).path).name
    if not filename or len(filename) < 5:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        extension = Path(urlparse(url).path).suffix
        filename = f"{url_hash}{extension}" if extension else f"{url_hash}.bin"
    return cache_dir / filename


def _is_cached(url: str, cache_dir: Path) -> tuple[bool, Path]:
    """URLに対応するファイルがキャッシュに存在するか確認する"""
    local_path = _get_cache_path(url, cache_dir)
    return local_path.is_file(), local_path


def _perform_download(url: str, target_path: Path) -> None:
    """実際のダウンロード処理を行う（進捗表示付き）"""
    logger.info(f"Downloading model from {url} to {target_path}")
    response = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    # ファイルサイズを取得
    total_size = int(response.headers.get("content-length", 0))

    with open(target_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=target_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_from_url(url: str, cache_dir: Path) -> Path:
    """URLからファイルをダウンロードします。"""
    cache_dir.mkdir(exist_ok=True, parents=True)
    is_cached, local_path = _is_cached(url, cache_dir)
    if not is_cached:
        _perform_download(url, local_path)
    return local_path.resolve()


def get_file_path(path_or_url: str, cache_dir: Path | str | None = None) -> Path:
    """パスまたはURLからローカルファイルパスを取得します。"""
    # cache_dir が None の場合はデフォルトパスを使用
    if cache_dir is None:
        cache_dir = DEFAULT_PATHS["cache_dir"]

    # 文字列の場合はPathオブジェクトに変換
    cache_dir_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
    assert isinstance(cache_dir_path, Path), "cache_dir_path must be a Path object"

    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return _download_from_url(path_or_url, cache_dir_path)
    else:
        return _get_local_file_path(path_or_url)


def _get_local_file_path(path: str) -> Path:
    """ローカルファイルパスを検証し、絶対パスを返します。"""
    local_path = Path(path)
    if local_path.exists():
        return local_path.resolve()
    raise FileNotFoundError(f"ローカルファイル '{path}' が見つかりません")


def extract_zip(zip_path: Path) -> Path:
    """
    ZIPファイルを解凍し、解凍先のディレクトリパスを返します。

    Args:
        zip_path: ZIPファイルのパス

    Returns:
        Path: 解凍先ディレクトリのパス

    Raises:
        RuntimeError: 解凍に失敗した場合
    """
    extract_dir = zip_path.parent / zip_path.stem
    try:
        logger.info(f"Extracting ZIP file: {zip_path} to {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir
    except Exception as e:
        raise RuntimeError(f"ZIPファイルの解凍に失敗しました: {e}") from e


def load_file(path_or_url: str) -> Path:
    """ファイルを取得し、ローカルパスを返します。"""
    cache_dir = DEFAULT_PATHS["cache_dir"]
    try:
        # cache_dirが文字列の場合は自動的にget_file_path内で変換される
        file_path = get_file_path(path_or_url, cache_dir)
        if file_path.suffix.lower() == ".zip":
            return extract_zip(file_path)
        return file_path
    except requests.RequestException as e:
        raise RuntimeError(f"URLからのダウンロードに失敗しました: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"ローカルファイルが見つかりません: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"'{path_or_url}' からのファイル取得に失敗しました。"
            "有効なローカルパス、または直接URLを指定してください。"
            f"エラー詳細: {e}"
        ) from e


def download_onnx_tagger_model(model_repo: str) -> tuple[Path, Path]:
    """WD-Taggerのモデルをダウンロードする"""
    # リポジトリ内のファイル一覧を取得
    repo_files = huggingface_hub.list_repo_files(model_repo)

    # CSVファイルを検索(最初に見つかったものを使用)
    csv_filename = next((f for f in repo_files if f.endswith(".csv")), WD_LABEL_FILENAME)

    csv_path = huggingface_hub.hf_hub_download(
        model_repo,
        csv_filename,
    )

    model_path = huggingface_hub.hf_hub_download(
        model_repo,
        WD_MODEL_FILENAME,
    )

    return Path(csv_path), Path(model_path)
