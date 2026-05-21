"""ONNX Runtime を使用する Tagger モデルの実装。"""

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
from PIL import Image

from ..core.base import ONNXBaseAnnotator
from ..core.config import config_registry
from ..core.types import TaskCapability, UnifiedAnnotationResult
from ..core.utils import logger


# E621カテゴリ番号の定義 (tagger-wrapper-lib からコピー)
# 参考: https://e621.net/wiki_pages/11262
# 0: general, 1: artist, 3: copyright, 4: character, 5: species, 6: invalid, 7: meta, 8: lore
class E621Categories:
    GENERAL = 0
    ARTIST = 1
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


class WDTagger(ONNXBaseAnnotator):
    """WD Tagger (ONNX版)"""

    rating_source_scheme = "danbooru4"

    def __init__(self, model_name: str):
        """WDTagger を初期化します。"""
        super().__init__(model_name=model_name)
        # カテゴリマッピング:WD-Taggerのカテゴリ番号を定義
        self.CATEGORY_MAPPING = {"rating": 9, "general": 0, "character": 4}
        # カテゴリIDとインデックス属性名の対応マップ
        # BaseAnnotator の _format_predictions_single で使用される
        self._category_attr_map = {
            "rating": "rating_indexes",
            "general": "general_indexes",
            "character": "character_indexes",
        }
        # 各カテゴリのインデックスリストを初期化
        self._init_empty_indexes()
        # tag_threshold の設定 (config_registry.get を使用、デフォルト値 0.35)
        self.tag_threshold = config_registry.get(self.model_name, "tag_threshold", 0.35)
        logger.info(f"Tag threshold set to: {self.tag_threshold}")

    def _load_tags(self) -> None:
        """タグ情報 (語彙) をロードし、カテゴリごとのインデックスを設定します。"""
        metadata_path = (
            self.components.get("metadata_path") or self.components.get("csv_path")
            if self.components
            else None
        )
        if not metadata_path:
            logger.error(
                "タグ情報ファイルパス (metadata_path/csv_path) が components に設定されていません。"
            )
            raise FileNotFoundError("タグ情報ファイルパスが見つかりません。")

        csv_path = metadata_path
        try:
            # ラベルファイルをpolarsで読み込み
            tags_df = pl.read_csv(csv_path)

            # タグ名を取得 (BaseAnnotator の self.all_tags に設定)
            self.all_tags = tags_df["name"].to_list()

            # カテゴリー情報を取得
            categories = tags_df["category"].to_list()

            # 全てのインデックスを初期化
            self._init_empty_indexes()

            # 各カテゴリのインデックスを抽出
            self._extract_category_indexes(categories)

            logger.info(f"WDタガータグ情報を読み込みました: 合計{len(self.all_tags)}個のタグ")
        except Exception as e:
            logger.error(f"タグ情報の読み込みに失敗しました ({csv_path}): {e}")
            # デフォルト値を設定
            self._init_empty_indexes()
            self.all_tags = []
            raise  # タグ情報読み込み失敗は致命的なのでエラーを送出

    def _init_empty_indexes(self) -> None:
        """各カテゴリのインデックスリストを空で初期化します。"""
        for attr_name in self._category_attr_map.values():
            setattr(self, attr_name, [])

    def _extract_category_indexes(self, categories: list[int]) -> None:
        """カテゴリリストから各カテゴリのインデックスを抽出します。"""
        for category_key, category_id in self.CATEGORY_MAPPING.items():
            # _category_attr_map にキーが存在するか確認
            if category_key in self._category_attr_map:
                attr_name = self._category_attr_map[category_key]
                indexes = [i for i, cat in enumerate(categories) if cat == category_id]
                setattr(self, attr_name, indexes)
            else:
                logger.warning(
                    f"カテゴリキー '{category_key}' に対応する属性名が _category_attr_map にありません。"
                )


class Z3D_E621Tagger(ONNXBaseAnnotator):
    """Z3D E621 Tagger (ONNX版)"""

    rating_source_scheme = "e6213"

    def __init__(self, model_name: str):
        """Z3D_E621Tagger を初期化します。"""
        super().__init__(model_name=model_name)
        # カテゴリIDとインデックス属性名の対応マップ
        # BaseAnnotator の _format_predictions_single で使用される
        self._category_attr_map = {
            "general": "general_indexes",
            "artist": "artist_indexes",
            "character": "character_indexes",
            "species": "species_indexes",
            "copyright": "copyright_indexes",
            "meta": "meta_indexes",  # meta, lore も追加
            "lore": "lore_indexes",
            "rating": "rating_indexes",
        }
        # レーティングタグ
        self._rating_tags = ["explicit", "questionable", "safe"]
        # 各カテゴリのインデックスリストを初期化
        self._init_empty_indexes()
        # tag_threshold の設定 (config_registry.get を使用、デフォルト値 0.35)
        self.tag_threshold = config_registry.get(self.model_name, "tag_threshold", 0.35)
        logger.info(f"Tag threshold set to: {self.tag_threshold}")

    def _load_tags(self) -> None:
        """Z3D_E621用のタグ情報 (語彙) をロードします。"""
        metadata_path = (
            self.components.get("metadata_path") or self.components.get("csv_path")
            if self.components
            else None
        )
        if not metadata_path:
            logger.error(
                "タグ情報ファイルパス (metadata_path/csv_path) が components に設定されていません。"
            )
            raise FileNotFoundError("タグ情報ファイルパスが見つかりません。")

        csv_path = metadata_path
        try:
            # CSVファイルの読み込みとタグ取得
            tags_df = pl.read_csv(csv_path)
            self.all_tags = tags_df["name"].to_list()
            self._init_empty_indexes()  # 各カテゴリのインデックスを初期化

            # カテゴリ処理
            self._process_categories(tags_df)

            logger.info(f"Z3D_E621タグ情報を読み込みました: 合計{len(self.all_tags)}個のタグ")

        except Exception as e:
            logger.error(f"タグ情報の読み込みに失敗しました ({csv_path}): {e}")
            # エラー時はデフォルト値を設定
            self._init_empty_indexes()
            self.all_tags = []
            raise  # タグ情報読み込み失敗は致命的なのでエラーを送出

    def _process_categories(self, tags_df: pl.DataFrame) -> None:
        """CSVからカテゴリ情報を処理します。"""
        if "category" in tags_df.columns:
            # カテゴリカラムがある場合
            categories = tags_df["category"].to_list()

            # 3つのステップでカテゴリ処理を行う
            self._extract_category_indexes(categories)  # 1. カテゴリに基づくインデックス抽出
            self._extract_rating_indexes()  # 2. レーティングタグの抽出
            self._handle_unclassified_tags(categories)  # 3. 未分類タグの処理 (引数に categories を追加)
        else:
            # カテゴリカラムがない場合はすべてを一般タグとして扱う
            self.general_indexes = list(range(len(self.all_tags)))

    def _init_empty_indexes(self) -> None:
        """各カテゴリのインデックスリストを空で初期化します。"""
        for attr_name in self._category_attr_map.values():
            setattr(self, attr_name, [])
        self.rating_indexes: list[int] = []  # rating_indexes も初期化

    def _extract_category_indexes(self, categories: list[int]) -> None:
        """カテゴリリストから各カテゴリのインデックスを抽出します。"""
        # E621Categories の値をキーとして使用
        e621_category_map = {
            E621Categories.GENERAL: "general_indexes",
            E621Categories.ARTIST: "artist_indexes",
            E621Categories.CHARACTER: "character_indexes",
            E621Categories.SPECIES: "species_indexes",
            E621Categories.COPYRIGHT: "copyright_indexes",
            E621Categories.META: "meta_indexes",
            E621Categories.LORE: "lore_indexes",
        }
        for category_id, attr_name in e621_category_map.items():
            indexes = [i for i, cat in enumerate(categories) if cat == category_id]
            setattr(self, attr_name, indexes)

    def _extract_rating_indexes(self) -> None:
        """レーティングタグのインデックスを抽出します。"""
        self.rating_indexes = [i for i, tag in enumerate(self.all_tags) if tag in self._rating_tags]

    def _handle_unclassified_tags(self, categories: list[int]) -> None:
        """未分類のタグを一般タグとして処理します。"""
        # 既知のカテゴリ ID をセットで保持 (E621Categories の値を使用)
        known_category_ids = {
            E621Categories.GENERAL,
            E621Categories.ARTIST,
            E621Categories.CHARACTER,
            E621Categories.SPECIES,
            E621Categories.COPYRIGHT,
            E621Categories.META,
            E621Categories.LORE,
            E621Categories.INVALID,  # INVALID も既知とする
        }

        # どのカテゴリにも属さないインデックスを抽出
        unclassified_indexes = [i for i, cat in enumerate(categories) if cat not in known_category_ids]

        # レーティングタグも除外
        rating_set = set(self.rating_indexes)
        unclassified_indexes = [i for i in unclassified_indexes if i not in rating_set]

        # 未分類のタグを一般タグに追加 (重複を避ける)
        current_general_indexes = getattr(self, "general_indexes", [])
        general_set = set(current_general_indexes)
        general_set.update(unclassified_indexes)
        self.general_indexes = list(general_set)


class CamieTagger(ONNXBaseAnnotator):
    """Camie Tagger (ONNX版)。"""

    onnx_model_filename: ClassVar[str] = "model_initial.onnx"
    onnx_metadata_filename: ClassVar[str] = "model_initial_metadata.json"
    onnx_metadata_extension: ClassVar[str] = ".json"
    rating_source_scheme: ClassVar[str] = "danbooru4"

    def __init__(self, model_name: str):
        """CamieTagger を初期化します。"""
        super().__init__(model_name=model_name)
        self._category_attr_map = {
            "rating": "rating_indexes",
            "general": "general_indexes",
            "character": "character_indexes",
        }
        self._init_empty_indexes()
        self.tag_threshold = config_registry.get(self.model_name, "tag_threshold", 0.325)
        logger.info(f"Camie tag threshold set to: {self.tag_threshold}")

    def _init_empty_indexes(self) -> None:
        """初期出力対象カテゴリのインデックスリストを空で初期化します。"""
        for attr_name in self._category_attr_map.values():
            setattr(self, attr_name, [])

    def _load_tags(self) -> None:
        """Camie JSON metadataからタグ名と対象カテゴリのインデックスをロードします。"""
        metadata_path = (
            self.components.get("metadata_path") or self.components.get("csv_path")
            if self.components
            else None
        )
        if not metadata_path:
            logger.error("Camie metadata file path is missing from ONNX components.")
            raise FileNotFoundError("Camie metadata file path not found.")

        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        idx_to_tag = metadata.get("idx_to_tag")
        tag_to_category = metadata.get("tag_to_category")
        if not isinstance(idx_to_tag, dict) or not isinstance(tag_to_category, dict):
            raise ValueError("Camie metadata must contain idx_to_tag and tag_to_category dictionaries.")

        self.all_tags = [str(idx_to_tag[str(i)]) for i in range(len(idx_to_tag))]
        self._init_empty_indexes()
        for index, tag_name in enumerate(self.all_tags):
            category = tag_to_category.get(tag_name)
            attr_name = self._category_attr_map.get(str(category))
            if attr_name:
                getattr(self, attr_name).append(index)

        logger.info(f"Camie metadata loaded: total_tags={len(self.all_tags)}")

    def _preprocess_images(self, images: list[Image.Image]) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
        """Camie公式ONNX経路に合わせ、黒padding + RGB + 0..1 NCHWで前処理します。"""
        target_size = self.target_size or (512, 512)
        image_size = target_size[0]
        results = []
        for image in images:
            img_rgb = image.convert("RGB") if image.mode in {"RGBA", "P"} or image.mode != "RGB" else image
            width, height = img_rgb.size
            aspect_ratio = width / height
            if aspect_ratio > 1:
                new_width = image_size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = image_size
                new_width = int(new_height * aspect_ratio)

            resized = img_rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
            padded = Image.new("RGB", (image_size, image_size), (0, 0, 0))
            padded.paste(resized, ((image_size - new_width) // 2, (image_size - new_height) // 2))
            array = np.asarray(padded, dtype=np.float32) / 255.0
            input_data = np.transpose(array, (2, 0, 1))
            results.append(np.expand_dims(input_data, axis=0).astype(np.float32))
        return results

    def _format_predictions_single(
        self, raw_output: np.ndarray[Any, np.dtype[Any]]
    ) -> UnifiedAnnotationResult:
        """Camie logitsにsigmoidを適用して統一結果へ整形します。"""
        from ..core.utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        probabilities = self._sigmoid(raw_output.astype(np.float32))
        predictions, error = self._validate_onnx_output(probabilities, self.all_tags, capabilities)
        if error:
            return error

        tags_with_probs = list(zip(self.all_tags, predictions, strict=True))
        category_scores = self._classify_tags_by_category(tags_with_probs)
        final_tags = self._filter_tags_by_threshold(category_scores, self.tag_threshold)
        ratings = self._extract_top_rating(category_scores, capabilities)

        return UnifiedAnnotationResult(
            model_name=self.model_name,
            capabilities=capabilities,
            tags=final_tags if TaskCapability.TAGS in capabilities else None,
            captions=None,
            scores=None,
            ratings=ratings,
            framework="onnx",
            raw_output={
                "category_scores": category_scores,
                "threshold": self.tag_threshold,
                "total_tags_count": len(self.all_tags),
            },
        )

    @staticmethod
    def _sigmoid(
        raw_output: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return 1.0 / (1.0 + np.exp(-raw_output))

    @staticmethod
    def _normalize_rating_label(raw_label: str) -> str:
        """Camie rating tags are stored as rating_general; expose danbooru4 labels."""
        normalized = ONNXBaseAnnotator._normalize_rating_label(raw_label)
        if normalized.startswith("rating_"):
            return normalized.removeprefix("rating_")
        return normalized
