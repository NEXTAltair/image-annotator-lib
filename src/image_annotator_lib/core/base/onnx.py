"""ONNX Runtime を使用するモデル用の基底クラス。"""

from abc import abstractmethod
from typing import Any, ClassVar, Self

import numpy as np
from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import OutOfMemoryError
from .. import onnx_runtime
from ..model_factory import ModelLoad
from ..types import (
    AnnotationErrorCode,
    ONNXComponents,
    RatingPrediction,
    TaskCapability,
    UnifiedAnnotationResult,
)
from ..utils import logger
from .local import LocalModelAnnotator, local_model_enter, local_model_exit


class ONNXBaseAnnotator(LocalModelAnnotator):
    """ONNX Runtime を使用するモデル用の基底クラス。"""

    onnx_model_filename: ClassVar[str] = "model.onnx"
    onnx_metadata_filename: ClassVar[str | None] = "selected_tags.csv"
    onnx_metadata_extension: ClassVar[str | None] = ".csv"

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # device 判定はローカル ML 系 base class の責務 (Issue #35 で BaseAnnotator から移譲)。
        # Issue #175: PyTorch の CUDA 可用性に加え、onnxruntime が CUDA EP を提供するかで判定する。
        self.requested_device = self._config.device
        self.device = onnx_runtime.determine_onnx_device(self._config.device, self.model_name)
        # ロード後に session.get_providers() から決める実際の実行先 (ロード前は None)。
        self.execution_device: str | None = None
        self.all_tags: list[str] = []
        self.target_size: tuple[int, int] | None = None
        self.is_nchw_expected: bool = False
        # components の型ヒントを具体的に指定
        self.components: ONNXComponents | None = None
        # Issue #162: ロード後の初期化 (_load_tags / _analyze_model_input_format) まで
        # 成功して初めて「再利用可能」とみなす。途中で失敗した components を
        # 次回呼び出しで使い回さないためのフラグ。
        self._prepared: bool = False
        # Issue #164: 動的 batch 軸でも出力が画像ごとに分割できないと分かったモデルは、
        # 以降まとめ直さず 1 枚ずつ推論する (毎回まとめ分を無駄にしない)。
        self._batching_disabled: bool = False
        # 直近の predict 1 回で呼んだ session.run の回数 (ログ用)。
        self._session_run_count: int = 0

    @local_model_enter
    def __enter__(self) -> Self:
        """ONNX モデルコンポーネントを準備する (準備済みなら再利用する)。

        Issue #162: `ModelLoad.load_onnx_components()` は既にロード済みの場合に `None` を
        返す。旧実装はその戻り値をそのまま代入していたため、セッションを保持したまま
        再入場すると `self.components` が None になり `_load_tags()` が失敗した。

        再利用は `_prepared` が True のときだけ行う。ロードは成功したが `_load_tags()` /
        `_analyze_model_input_format()` が失敗した場合に中途半端な components を
        使い回すと、タグ未ロードや target_size 未設定のまま推論してしまうため。
        """
        try:
            if self.model_path is None:
                raise ValueError(f"モデル '{self.model_name}' の model_path が設定されていません。")

            if self._prepared and self.components and self.components.get("session") is not None:
                logger.debug(f"ONNX components already prepared for '{self.model_name}'; reusing")
                # LRU の最終使用時刻だけ更新して再利用する。
                self._load_components()
                return self

            logger.info(f"Loading/Restoring ONNX components: model='{self.model_path}'")
            loaded = self._load_components()

            if loaded is None:
                raise RuntimeError(f"ONNX モデル '{self.model_name}' のロードに失敗しました。")

            self.components = loaded
            self.execution_device = onnx_runtime.session_execution_device(loaded["session"])
            self._load_tags()
            self._analyze_model_input_format()
            self._prepared = True
            # LRU 退避 / 明示解放から実体も解放できるようにする (Issue #162)。
            ModelLoad.register_component_releaser(
                self.model_name, self._release_retained_components, owner=self
            )

        except OutOfMemoryError as e:
            raise e
        except Exception as e:
            logger.exception(f"ONNXモデル {self.model_name} の準備中にエラーが発生: {e}")
            raise

        return self

    def _sync_execution_device(self, session: Any) -> None:
        """推論後に session の provider を読み直し、実行先と状態台帳を実際に合わせる (Issue #175)。"""
        previous = self.execution_device
        actual = onnx_runtime.follow_session_providers(self.model_name, session, previous)
        if actual == previous:
            return
        self.execution_device = actual
        if previous is not None and ModelLoad._get_model_state(self.model_name):
            ModelLoad._update_model_state(self.model_name, actual, "loaded")

    def _load_components(self) -> ONNXComponents | None:
        """`ModelLoad.load_onnx_components` を本クラスの設定で呼ぶ薄いヘルパー。"""
        return ModelLoad.load_onnx_components(
            self.model_name,
            str(self.model_path),
            self.device,
            model_filename=self.onnx_model_filename,
            metadata_filename=self.onnx_metadata_filename,
            metadata_extension=self.onnx_metadata_extension,
        )

    @local_model_exit
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストを抜ける。セッションは保持したままにする (Issue #162)。

        旧実装はここで `ModelLoad.release_model_components()` を呼び InferenceSession を
        完全破棄していたため、`annotate()` 呼び出しごとに CUDA EP を含むセッション再構築が
        走っていた (実測 1 チャンク 3.842 秒のうちロード 1.704 秒 + 破棄 0.607 秒)。
        Transformers 系が `cache_to_main_memory()` で退避に留めるのと非対称でもあった。

        解放は `ModelLoad` の LRU (`_release_model_state` 経由で
        `_release_retained_components` が呼ばれる)、または呼び出し側の明示的な
        `ModelLoad.release_model(model_name)` に委ねる。
        """
        logger.debug(f"Exiting context for ONNX model '{self.model_name}' (exception: {exc_type})")
        if exc_type:
            logger.error(f"ONNX モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}")

    @abstractmethod
    def _load_tags(self) -> None:
        """タグ情報 (語彙) をロードし、必要に応じてカテゴリインデックスを設定します (サブクラスで実装)。"""
        raise NotImplementedError("ONNX サブクラスは _load_tags を実装する必要があります。")

    def _extract_category_tags(
        self, attr_name: str, tags_with_probs: list[tuple[str, float]]
    ) -> dict[str, float]:
        """カテゴリータグを抽出するヘルパー関数 (ONNX タガー用)。"""
        category_tags: dict[str, float] = {}
        indexes = getattr(self, attr_name, [])
        all_tags_list = getattr(self, "all_tags", [])
        for i in indexes:
            if 0 <= i < len(tags_with_probs):
                tag_name, prob = tags_with_probs[i]
                category_tags[tag_name] = prob
            else:
                logger.warning(f"インデックス {i} が範囲外です (タグ総数: {len(all_tags_list)})。")
        return category_tags

    def _to_probabilities(
        self, raw_output: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """ONNX生出力を確率値へ変換する。

        確率を直接出力するモデル (WDTagger / Z3D 等) では恒等変換 (既定)。
        logit を出力するモデル (CamieTagger 等) はサブクラスで sigmoid 等を適用する。

        Args:
            raw_output: ONNX推論の生出力配列。

        Returns:
            確率値の配列。
        """
        return raw_output

    def _format_predictions_single(
        self, raw_output: np.ndarray[Any, np.dtype[Any]]
    ) -> UnifiedAnnotationResult:
        """ONNX生出力を統一UnifiedAnnotationResultにフォーマットする。

        Args:
            raw_output: ONNX推論の生出力配列。

        Returns:
            フォーマット済みのUnifiedAnnotationResult。
        """
        from ..utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        all_tags_list = getattr(self, "all_tags", [])
        threshold = getattr(self, "tag_threshold", 0.35)

        # バリデーション (生出力 → 確率変換を挟む)
        predictions, error = self._validate_onnx_output(
            self._to_probabilities(raw_output), all_tags_list, capabilities
        )
        if error:
            return error
        if predictions is None:
            raise RuntimeError("ONNX output validation returned no predictions or error.")

        # カテゴリ別にタグを分類
        tags_with_probs = list(zip(all_tags_list, predictions, strict=True))
        category_scores = self._classify_tags_by_category(tags_with_probs)

        # 閾値フィルタリング
        final_tags = self._filter_tags_by_threshold(category_scores, threshold)

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
                "predictions": predictions.tolist(),
                "category_scores": category_scores,
                "threshold": threshold,
                "total_tags_count": len(all_tags_list),
            },
        )

    def _validate_onnx_output(
        self,
        raw_output: np.ndarray[Any, np.dtype[Any]],
        all_tags_list: list[str],
        capabilities: set,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]] | None, UnifiedAnnotationResult | None]:
        """ONNX出力のバリデーションを行う。

        Args:
            raw_output: ONNX推論の生出力。
            all_tags_list: タグ候補リスト。
            capabilities: モデルのケイパビリティ。

        Returns:
            (predictions, error) のタプル。バリデーション成功時はerrorがNone。
        """
        if not all_tags_list:
            logger.warning("タグ候補リスト (all_tags) がロードされていません。")
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error="タグ候補リストが未ロード",
                framework="onnx",
            )

        if not isinstance(raw_output, np.ndarray):
            logger.error(f"予期しない生出力型: {type(raw_output)}")
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error=f"予期しない生出力型: {type(raw_output)}",
                framework="onnx",
            )

        predictions: np.ndarray[Any, np.dtype[np.float64]]
        if raw_output.ndim == 2 and raw_output.shape[0] == 1:
            predictions = raw_output[0].astype(float)
        elif raw_output.ndim == 1:
            predictions = raw_output.astype(float)
        else:
            logger.error(f"予期しない生出力形状: {raw_output.shape}")
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error=f"予期しない生出力形状: {raw_output.shape}",
                framework="onnx",
            )

        if len(all_tags_list) != len(predictions):
            logger.error(
                f"タグ候補リスト数 ({len(all_tags_list)}) と予測数 ({len(predictions)}) が一致しません。"
            )
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error=f"タグ数と予測数の不一致: {len(all_tags_list)} != {len(predictions)}",
                framework="onnx",
            )

        return predictions, None

    def _classify_tags_by_category(
        self, tags_with_probs: list[tuple[str, float]]
    ) -> dict[str, dict[str, float]]:
        """タグをカテゴリ別に分類する。

        Args:
            tags_with_probs: (タグ名, 確率) のリスト。

        Returns:
            カテゴリ名→{タグ名: 確率} の辞書。
        """
        category_scores: dict[str, dict[str, float]] = {}

        category_map = getattr(self, "_category_attr_map", None)
        if category_map is None:
            logger.warning(
                "_category_attr_map がサブクラスで定義されていません。カテゴリ分類なしでフォーマットします。"
            )
            category_scores["general"] = {tag: float(prob) for tag, prob in tags_with_probs}
        else:
            for category_key, attr_name in category_map.items():
                category_tags = self._extract_category_tags(attr_name, tags_with_probs)
                if category_tags:
                    category_scores[category_key] = category_tags

            if "rating" in category_scores and "ratings" not in category_scores:
                category_scores["ratings"] = category_scores.pop("rating")

        return category_scores

    def _extract_top_rating(
        self, category_scores: dict[str, dict[str, float]], capabilities: set[TaskCapability]
    ) -> list[RatingPrediction] | None:
        """Extract top-1 model-native rating prediction from category scores."""
        if TaskCapability.RATINGS not in capabilities:
            return None

        rating_scores = category_scores.get("ratings")
        if not rating_scores:
            return None

        raw_label, confidence = max(rating_scores.items(), key=lambda item: item[1])
        source_scheme = getattr(self, "rating_source_scheme", "unknown")
        return [
            RatingPrediction(
                raw_label=self._normalize_rating_label(raw_label),
                confidence_score=float(confidence),
                source_scheme=source_scheme,
            )
        ]

    @staticmethod
    def _normalize_rating_label(raw_label: str) -> str:
        """Normalize adapter-local rating label spelling without consumer mapping."""
        if ":" in raw_label:
            raw_label = raw_label.split(":", 1)[1]
        return raw_label.strip().lower().replace(" ", "_")

    @staticmethod
    def _filter_tags_by_threshold(
        category_scores: dict[str, dict[str, float]], threshold: float
    ) -> list[str]:
        """閾値を超えるタグを信頼度順でフィルタリングする。

        Args:
            category_scores: カテゴリ別スコア辞書。
            threshold: フィルタ閾値。

        Returns:
            信頼度降順のタグ名リスト。
        """
        filtered_tags = []
        for category, tag_dict in category_scores.items():
            if category in {"rating", "ratings"}:
                continue
            for tag, confidence in tag_dict.items():
                if confidence >= threshold:
                    filtered_tags.append((tag, confidence))

        filtered_tags.sort(key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in filtered_tags]

    def _generate_tags_single(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (ONNX タガー用)。"""
        tags = []
        if not formatted_output or "error" in formatted_output:
            return []

        # tag_threshold はサブクラスで設定される想定
        threshold = getattr(self, "tag_threshold", 0.35)

        for category, tag_dict in formatted_output.items():
            if category == "error":
                continue
            for tag, confidence in tag_dict.items():
                conf_value = (
                    confidence["confidence"]
                    if isinstance(confidence, dict) and "confidence" in confidence
                    else confidence
                )

                if isinstance(conf_value, (float)) and conf_value >= threshold:
                    tags.append((tag, float(conf_value)))

        unique_tags: dict[str, float] = {}
        for tag, conf in tags:
            if tag not in unique_tags or conf > unique_tags[tag]:
                unique_tags[tag] = conf

        return [tag for tag, _ in sorted(unique_tags.items(), key=lambda x: x[1], reverse=True)]

    def _analyze_model_input_format(self) -> None:
        """モデル入力形式を分析し、ターゲットサイズと次元形式を判定･保存する"""
        if not self.components or "session" not in self.components or self.components["session"] is None:
            raise RuntimeError("ONNX セッションがロードされていません。")
        session = self.components["session"]
        input_shape = session.get_inputs()[0].shape

        target_size: tuple[int, int] | None = None
        is_nchw = False
        if len(input_shape) == 4:
            if (
                isinstance(input_shape[1], int)
                and input_shape[1] == 3
                and isinstance(input_shape[2], int)
                and isinstance(input_shape[3], int)
            ):
                target_size = (input_shape[2], input_shape[3])
                is_nchw = True
            elif (
                isinstance(input_shape[3], int)
                and input_shape[3] == 3
                and isinstance(input_shape[1], int)
                and isinstance(input_shape[2], int)
            ):
                target_size = (input_shape[1], input_shape[2])
                is_nchw = False
            else:
                if isinstance(input_shape[1], int) and isinstance(input_shape[2], int):
                    logger.warning(
                        f"モデル {self.model_name} の不明な入力形状フォーマット: {input_shape}。ターゲットサイズとしてNHWC (インデックス 1, 2) を想定します。"
                    )
                    target_size = (input_shape[1], input_shape[2])
                    is_nchw = False

        if target_size is None:
            raise ValueError(f"入力形状 {input_shape} から有効なターゲットサイズ (H, W) を決定できません。")

        self.target_size = target_size
        self.is_nchw_expected = is_nchw

        logger.debug(
            f"モデル {self.model_name} の入力形状: {input_shape}, ターゲットサイズ: {self.target_size}, NCHW形式: {self.is_nchw_expected}"
        )

    def _preprocess_images(self, images: list[Image.Image]) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
        """画像バッチを前処理します。各画像を個別に処理して結果をリストで返します。"""
        if self.target_size is None:
            raise ValueError(f"モデル {self.model_name} の target_size が設定されていません。")

        results = []
        for image in images:
            if image.mode == "RGBA":
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                canvas.paste(image, mask=image.split()[3])
                img_rgb = canvas
            elif image.mode != "RGB":
                img_rgb = image.convert("RGB")
            else:
                img_rgb = image

            width, height = img_rgb.size
            max_dim = max(width, height)
            pad_width = (max_dim - width) // 2
            pad_height = (max_dim - height) // 2
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded.paste(img_rgb, (pad_width, pad_height))

            resized = padded.resize(self.target_size, Image.Resampling.LANCZOS)
            img_array = np.array(resized, dtype=np.float32)[:, :, ::-1]

            if self.is_nchw_expected:
                input_data = np.transpose(img_array, (2, 0, 1))
            else:
                input_data = img_array

            input_data = np.expand_dims(input_data, axis=0)
            results.append(input_data.astype(np.float32))
        return results

    def _inference_batch_size(self) -> int:
        """設定 ``inference_batch_size`` (未設定なら ``DEFAULT_INFERENCE_BATCH_SIZE``) を返す。"""
        value = getattr(self._config, "inference_batch_size", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return onnx_runtime.DEFAULT_INFERENCE_BATCH_SIZE

    @staticmethod
    def _is_dynamic_axis(dim: Any) -> bool:
        """ONNX の shape 要素が動的か (文字列 = dim_param、None = 記載なし)。静的なのは int。"""
        return dim is None or isinstance(dim, str)

    def _can_batch(self, session: Any) -> bool:
        """入力の batch 軸が動的で、出力の batch 軸も固定でなく、まとめ推論を試せるか。

        静的 batch (1 や N) のモデルは、1 枚ずつ投げる従来の動作を保つ。出力の batch 軸が
        固定 (例: 入力は動的でも内部で batch 軸を潰す) と宣言されていれば、分割できないので試さない。
        """
        if self._batching_disabled:
            return False
        input_shape = session.get_inputs()[0].shape
        if not isinstance(input_shape, list | tuple) or not input_shape:
            return False
        if not self._is_dynamic_axis(input_shape[0]):
            return False
        output_shape = session.get_outputs()[0].shape
        if isinstance(output_shape, list | tuple):
            return bool(output_shape) and self._is_dynamic_axis(output_shape[0])
        return True

    @staticmethod
    def _is_stackable(group: list[np.ndarray[Any, np.dtype[Any]]]) -> bool:
        """各配列が batch 軸 1 の同一 shape / dtype で、axis=0 で連結できるか。"""
        first = group[0]
        if first.ndim < 1 or first.shape[0] != 1:
            return False
        return all(item.shape == first.shape and item.dtype == first.dtype for item in group)

    def _session_run(
        self,
        session: Any,
        input_name: str,
        output_name: str,
        input_data: np.ndarray[Any, np.dtype[Any]],
    ) -> Any:
        """session.run を 1 回実行し、provider を読み直す。OOM は ``OutOfMemoryError`` に変換する。"""
        try:
            raw_output = session.run([output_name], {input_name: input_data})
            self._session_run_count += 1
            self._sync_execution_device(session)
            return raw_output[0]
        except Exception as e:
            if "Failed to allocate memory" in str(e) or "out of memory" in str(e).lower():
                error_message = f"ONNX Runtime メモリ不足: モデル {self.model_name} の推論中"
                logger.error(error_message)
                logger.error(f"元のONNX Runtimeエラー: {e}")
                # Issue #162: `BaseAnnotator.predict()` は OutOfMemoryError を握って
                # エラー結果に変換するため、`__exit__` からは正常終了に見える。
                # セッションを保持したままだと後続チャンクが同じ逼迫した
                # セッションを再利用し続けるので、ここで明示的に無効化して
                # 次回クリーンに再ロードさせる (GC / CUDA キャッシュ解放も走る)。
                self._discard_components()
                raise OutOfMemoryError(error_message) from e
            logger.exception(f"ONNX Runtime エラー: モデル {self.model_name} の推論中: {e}")
            raise

    def _run_batched_group(
        self,
        session: Any,
        input_name: str,
        output_name: str,
        group: list[np.ndarray[Any, np.dtype[Any]]],
    ) -> list[np.ndarray[Any, np.dtype[Any]]] | None:
        """複数画像を連結して 1 回で推論し、画像ごとの出力 ([1, ...]) に分割して返す。

        出力の先頭軸が画像数と一致しない (分割できない) 場合は None を返す。
        """
        batch = np.concatenate(group, axis=0)
        output = self._session_run(session, input_name, output_name, batch)
        if isinstance(output, np.ndarray) and output.ndim >= 1 and output.shape[0] == len(group):
            return [output[i : i + 1] for i in range(len(group))]
        shape = getattr(output, "shape", None)
        logger.warning(
            f"モデル {self.model_name}: {len(group)} 枚をまとめた出力 (shape={shape}) が画像ごとに分割できないため、"
            "以降は 1 枚ずつ推論します。"
        )
        self._batching_disabled = True
        return None

    def _run_inference(
        self, processed: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> list[np.ndarray[Any, np.dtype[Any]]]:
        """前処理済みの各画像に対して ONNX 推論を実行し、画像ごとの出力を入力順に返します。

        動的 batch 軸のモデルでは、``inference_batch_size`` 枚ごとに連結して 1 回の
        ``session.run`` で推論する (Issue #164)。静的 batch のモデル、shape が揃わない入力、
        出力が分割できないモデルは 1 枚ずつ推論する。
        """
        if not self.components or "session" not in self.components or self.components["session"] is None:
            raise RuntimeError("ONNX セッションがロードされていません。")
        session = self.components["session"]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        group_size = self._inference_batch_size() if self._can_batch(session) else 1
        results: list[np.ndarray[Any, np.dtype[Any]]] = []
        index = 0
        while index < len(processed):
            group = processed[index : index + group_size]
            index += len(group)
            if len(group) > 1 and self._is_stackable(group):
                split = self._run_batched_group(session, input_name, output_name, group)
                if split is not None:
                    results.extend(split)
                    continue
                # 分割できなかった回は、同じ画像を 1 枚ずつ投げ直す。
                group_size = 1
            results.extend(self._session_run(session, input_name, output_name, item) for item in group)
        return results

    def _execute_pipeline(self, images: list[Image.Image]) -> list[Any]:
        """画像を ``inference_batch_size`` 枚ずつ前処理・推論・整形する (Issue #164)。

        前処理済み配列と連結した入力は 1 回分だけ保持し、全画像分を同時に持たない。
        推論中の OOM ではモデルを解放し (``_session_run``)、完了済みの回の結果は保持して、
        残りの画像を ``OUT_OF_MEMORY`` のエラー結果にする。再試行はしない。
        結果の順序と数は入力画像と一致する。
        """
        total = len(images)
        chunk_size = self._inference_batch_size()
        self._session_run_count = 0
        outputs: list[Any] = []
        index = 0
        logger.info(f"ONNX 推論開始: model='{self.model_name}', 画像数={total}, 推論枚数上限={chunk_size}")
        while index < total:
            chunk = images[index : index + chunk_size]
            try:
                outputs.extend(super()._execute_pipeline(chunk))
            except OutOfMemoryError as error:
                message = f"推論中にメモリ不足 (OOM): {error}"
                logger.error(
                    f"モデル '{self.model_name}': {message}。完了済み {index} 枚の結果を保持し、"
                    f"残り {total - index} 枚をエラー結果にします。"
                )
                outputs.extend(
                    self._create_error_results(
                        images[index:], message, error_code=AnnotationErrorCode.OUT_OF_MEMORY
                    )
                )
                break
            index += len(chunk)
        logger.info(
            f"ONNX 推論完了: model='{self.model_name}', 実行先={self.execution_device}, "
            f"成功={min(index, total)}/{total}, session.run 回数={self._session_run_count}"
        )
        return outputs

    def _format_predictions(
        self, raw_outputs: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> list[UnifiedAnnotationResult]:
        """バッチ出力結果を統一バリデーションスキーマでフォーマット"""
        result_list = []
        for raw_output in raw_outputs:
            formatted = self._format_predictions_single(raw_output)
            # Issue #175: 要求ではなく session の provider から決めた実際の実行先を結果に載せる。
            formatted.execution_device = self.execution_device
            result_list.append(formatted)
        return result_list

    def _generate_tags(self, formatted_output: UnifiedAnnotationResult) -> list[str]:
        """統一バリデーションスキーマからタグリストを生成"""
        if isinstance(formatted_output, UnifiedAnnotationResult):
            return formatted_output.tags or []
        return []
