"""model_installer (明示モデルインストール API) のユニットテスト (LoRAIro Issue #754)"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.exceptions.errors import ModelInstallCancelledError, ModelNotFoundError
from image_annotator_lib.model_installer import (
    ModelInstallProgress,
    _build_snapshot_tqdm_class,
    _ByteProgressAggregator,
    _hf_patterns_for_class,
    _transformers_ignore_patterns,
    install_model,
    is_model_installed,
)


@pytest.fixture
def fake_paths(tmp_path):
    """cache_dir を tmp_path に差し替える。"""
    paths = {"cache_dir": tmp_path / "models"}
    with patch("image_annotator_lib.model_installer.DEFAULT_PATHS", paths):
        yield paths


def _patch_config(config: dict):
    """model_installer が参照する config_registry を差し替える。

    config_registry はプロキシオブジェクトで属性単位の patch teardown が
    効かないため、プロキシごと Mock に差し替える。
    """
    registry = MagicMock()
    registry.get_all_config.return_value = config
    return patch("image_annotator_lib.model_installer.config_registry", registry)


class TestModelInstallProgress:
    def test_percentage_normal(self):
        progress = ModelInstallProgress(model_name="m", downloaded_bytes=50, total_bytes=200)
        assert progress.percentage == 25

    def test_percentage_unknown_total_returns_zero(self):
        progress = ModelInstallProgress(model_name="m", downloaded_bytes=50, total_bytes=0)
        assert progress.percentage == 0

    def test_percentage_capped_at_100(self):
        progress = ModelInstallProgress(model_name="m", downloaded_bytes=300, total_bytes=200)
        assert progress.percentage == 100


class TestIsModelInstalled:
    def test_unknown_model_returns_true(self, fake_paths):
        with _patch_config({}):
            assert is_model_installed("openai/gpt-4o") is True

    def test_model_without_model_path_returns_true(self, fake_paths):
        with _patch_config({"some_model": {"class": "WDTagger"}}):
            assert is_model_installed("some_model") is True

    def test_known_model_without_marker_returns_false(self, fake_paths):
        config = {"wd-vit-tagger-v3": {"model_path": "SmilingWolf/wd-vit-tagger-v3", "class": "WDTagger"}}
        with _patch_config(config):
            assert is_model_installed("wd-vit-tagger-v3") is False

    def test_install_then_installed(self, fake_paths):
        config = {"wd-vit-tagger-v3": {"model_path": "SmilingWolf/wd-vit-tagger-v3", "class": "WDTagger"}}
        with (
            _patch_config(config),
            patch("huggingface_hub.snapshot_download") as mock_snapshot,
        ):
            install_model("wd-vit-tagger-v3")
            assert mock_snapshot.call_count == 1
            assert is_model_installed("wd-vit-tagger-v3") is True


class TestInstallModel:
    def test_unknown_model_raises(self, fake_paths):
        with _patch_config({}), pytest.raises(ModelNotFoundError):
            install_model("no_such_model")

    def test_onnx_tagger_uses_allow_patterns(self, fake_paths):
        config = {"wd-vit-tagger-v3": {"model_path": "SmilingWolf/wd-vit-tagger-v3", "class": "WDTagger"}}
        with (
            _patch_config(config),
            patch("huggingface_hub.snapshot_download") as mock_snapshot,
        ):
            install_model("wd-vit-tagger-v3")
        _, kwargs = mock_snapshot.call_args
        assert kwargs["allow_patterns"] == ["model.onnx", "*.csv"]
        assert kwargs["ignore_patterns"] is None

    def test_cancel_before_download_raises_without_marker(self, fake_paths):
        config = {"wd-vit-tagger-v3": {"model_path": "SmilingWolf/wd-vit-tagger-v3", "class": "WDTagger"}}
        cancel_event = threading.Event()
        cancel_event.set()
        with (
            _patch_config(config),
            patch("huggingface_hub.snapshot_download") as mock_snapshot,
            pytest.raises(ModelInstallCancelledError),
        ):
            install_model("wd-vit-tagger-v3", cancel_event=cancel_event)
        assert mock_snapshot.call_count == 0
        with _patch_config(config):
            assert is_model_installed("wd-vit-tagger-v3") is False

    def test_clip_model_installs_url_and_base_model(self, fake_paths, tmp_path):
        config = {
            "ImprovedAesthetic": {
                "model_path": "https://example.com/sac+logos+ava1-l14-linearMSE.pth",
                "base_model": "openai/clip-vit-large-patch14",
                "class": "ImprovedAesthetic",
            }
        }
        received: list[ModelInstallProgress] = []
        with (
            _patch_config(config),
            patch("requests.get") as mock_get,
            patch("huggingface_hub.snapshot_download") as mock_snapshot,
            patch(
                "image_annotator_lib.model_installer._transformers_ignore_patterns",
                return_value=["*.h5"],
            ),
        ):
            mock_get.return_value.headers = {"content-length": "10"}
            mock_get.return_value.iter_content.return_value = iter([b"0123456789"])
            install_model("ImprovedAesthetic", progress_callback=received.append)

        # URL リソースがキャッシュ規約どおりに保存される
        cached = fake_paths["cache_dir"] / "sac+logos+ava1-l14-linearMSE.pth"
        assert cached.read_bytes() == b"0123456789"
        # base_model の snapshot も実行される
        args, kwargs = mock_snapshot.call_args
        assert args[0] == "openai/clip-vit-large-patch14"
        assert kwargs["ignore_patterns"] == ["*.h5"]
        # byte 進捗が通知される
        assert received[-1].downloaded_bytes == 10
        assert received[-1].total_bytes == 10
        assert received[-1].percentage == 100

    def test_url_download_cancel_leaves_no_cache_file(self, fake_paths):
        config = {
            "WaifuAesthetic": {
                "model_path": "https://example.com/aes-B32-v0.pth",
                "class": "WaifuAesthetic",
            }
        }
        cancel_event = threading.Event()

        def chunks():
            yield b"01234"
            cancel_event.set()
            yield b"56789"

        with (
            _patch_config(config),
            patch("requests.get") as mock_get,
            pytest.raises(ModelInstallCancelledError),
        ):
            mock_get.return_value.headers = {"content-length": "10"}
            mock_get.return_value.iter_content.return_value = chunks()
            install_model("WaifuAesthetic", cancel_event=cancel_event)

        cache_dir = fake_paths["cache_dir"]
        assert not (cache_dir / "aes-B32-v0.pth").exists()
        assert not (cache_dir / "aes-B32-v0.pth.part").exists()

    def test_model_without_model_path_skips(self, fake_paths):
        with (
            _patch_config({"odd_model": {"class": "WDTagger"}}),
            patch("huggingface_hub.snapshot_download") as mock_snapshot,
        ):
            install_model("odd_model")
        assert mock_snapshot.call_count == 0


class TestHfPatternsForClass:
    def test_anime_rating_uses_variant_files(self):
        config = {"model_variant": "mobilenetv3_sce_dist"}
        patterns = _hf_patterns_for_class(config, "AnimeRatingAnnotator")
        assert patterns.allow_patterns == [
            "mobilenetv3_sce_dist/model.onnx",
            "mobilenetv3_sce_dist/meta.json",
        ]

    def test_camie_tagger_uses_initial_files(self):
        patterns = _hf_patterns_for_class({}, "CamieTagger")
        assert patterns.allow_patterns == ["model_initial.onnx", "*.json"]

    def test_transformers_class_falls_back_to_snapshot(self):
        patterns = _hf_patterns_for_class({}, "BLIPTagger")
        assert patterns.allow_patterns is None
        assert patterns.transformers_weights is True


class TestTransformersIgnorePatterns:
    def test_safetensors_repo_ignores_duplicate_weights(self):
        with patch("huggingface_hub.list_repo_files", return_value=["model.safetensors", "config.json"]):
            patterns = _transformers_ignore_patterns("org/repo")
        assert "*.bin" in patterns
        assert "*.pth" in patterns

    def test_bin_only_repo_keeps_bin(self):
        with patch("huggingface_hub.list_repo_files", return_value=["pytorch_model.bin", "config.json"]):
            patterns = _transformers_ignore_patterns("org/repo")
        assert "*.bin" not in patterns
        assert "*.h5" in patterns

    def test_listing_failure_falls_back_to_static(self):
        with patch("huggingface_hub.list_repo_files", side_effect=OSError("offline")):
            patterns = _transformers_ignore_patterns("org/repo")
        assert "*.bin" not in patterns
        assert "*.h5" in patterns


class TestSnapshotProgressTqdm:
    """snapshot_download の tqdm_class フック挙動 (byte バーのみ追跡) を検証する。"""

    def _aggregator(self, received, cancel_event=None):
        return _ByteProgressAggregator("m", received.append, cancel_event)

    def test_bytes_bar_reports_progress(self):
        received: list[ModelInstallProgress] = []
        tqdm_cls = _build_snapshot_tqdm_class(self._aggregator(received))

        # hf の bytes_progress 相当: unit="B" で作成され total が外部加算される
        bar = tqdm_cls(desc="Downloading", total=0, initial=0, unit="B", unit_scale=True)
        bar.total += 100  # _AggregatedTqdm.__init__ 相当
        bar.refresh()
        bar.update(40)
        bar.update(60)

        assert received[-1].downloaded_bytes == 100
        assert received[-1].total_bytes == 100
        assert received[-1].percentage == 100

    def test_file_count_bar_is_ignored(self):
        received: list[ModelInstallProgress] = []
        tqdm_cls = _build_snapshot_tqdm_class(self._aggregator(received))

        # thread_map の外側バー相当 (unit 指定なし): byte 集計に混入しない
        bar = tqdm_cls(total=3, desc="Fetching 3 files")
        bar.update(1)
        assert received == []

    def test_update_raises_when_cancelled(self):
        received: list[ModelInstallProgress] = []
        cancel_event = threading.Event()
        tqdm_cls = _build_snapshot_tqdm_class(self._aggregator(received, cancel_event))
        bar = tqdm_cls(desc="Downloading", total=0, initial=0, unit="B", unit_scale=True)

        cancel_event.set()
        with pytest.raises(ModelInstallCancelledError):
            bar.update(10)


class TestCodexReviewFixes:
    """PR #151 Codex P2 指摘への対応 (stale marker / local path / cancel 前倒し)"""

    def test_stale_marker_with_different_model_path_returns_false(self, fake_paths):
        """config の model_path 付け替え後は古い marker を stale として拒否する"""
        config_v1 = {"m": {"model_path": "org/repo-v1", "class": "WDTagger"}}
        config_v2 = {"m": {"model_path": "org/repo-v2", "class": "WDTagger"}}
        with _patch_config(config_v1), patch("huggingface_hub.snapshot_download"):
            install_model("m")
            assert is_model_installed("m") is True
        with _patch_config(config_v2):
            assert is_model_installed("m") is False

    def test_malformed_marker_returns_false(self, fake_paths):
        """壊れた marker ファイルは未インストール扱いにする"""
        config = {"m": {"model_path": "org/repo", "class": "WDTagger"}}
        marker_dir = fake_paths["cache_dir"] / "install_markers"
        marker_dir.mkdir(parents=True)
        (marker_dir / "m.json").write_text("{not json", encoding="utf-8")
        with _patch_config(config):
            assert is_model_installed("m") is False

    def test_local_path_model_skips_download(self, fake_paths, tmp_path):
        """ローカルパス指定の model_path は DL せず marker のみ書く"""
        local_model = tmp_path / "local_model.onnx"
        local_model.write_bytes(b"onnx")
        config = {"m": {"model_path": str(local_model), "class": "WDTagger"}}
        with (
            _patch_config(config),
            patch("huggingface_hub.snapshot_download") as mock_snapshot,
        ):
            install_model("m")
            assert mock_snapshot.call_count == 0
            assert is_model_installed("m") is True

    def test_cancel_prevents_repo_listing_for_transformers_class(self, fake_paths):
        """キャンセル済みなら transformers 系の repo 一覧取得 (network) にも出ない"""
        config = {"blip": {"model_path": "Salesforce/blip", "class": "BLIPTagger"}}
        cancel_event = threading.Event()
        cancel_event.set()
        with (
            _patch_config(config),
            patch("huggingface_hub.list_repo_files") as mock_list,
            patch("huggingface_hub.snapshot_download") as mock_snapshot,
            pytest.raises(ModelInstallCancelledError),
        ):
            install_model("blip", cancel_event=cancel_event)
        assert mock_list.call_count == 0
        assert mock_snapshot.call_count == 0
