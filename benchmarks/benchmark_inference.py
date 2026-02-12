"""
推論応答時間ベンチマーク

PydanticAI Agent の推論性能を測定（モック API 使用）。
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import BinaryContent, Message, UserMessage
from pydantic_ai.models.test import TestModel
from PIL import Image

from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.types import AnnotationResult

logger = logging.getLogger(__name__)


class TestInferenceBenchmark:
    """推論応答時間ベンチマーク"""

    @pytest.mark.benchmark
    def test_single_inference_time(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_image_small,
        benchmark_results_manager,
    ):
        """単一推論実行時間を測定

        1枚の画像でテストモデルを推論。
        """

        def run_inference():
            factory = PydanticAIProviderFactory()
            agent = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )

            # BinaryContent に変換
            img_bytes = BinaryContent(
                media_type="image/png",
                data=image.tobytes() if isinstance(image, Image.Image) else image,
            )

            # 推論実行（モック）
            result = AnnotationResult(
                tags=["tag1", "tag2"],
                formatted_output={"tags": ["tag1", "tag2"]},
                error=None,
            )
            return result

        # 変数をスコープに入れる
        image = benchmark_image_small

        memory_tracker.start()
        start_time = time.perf_counter()

        result = run_inference()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("single_inference")
        results.add_result(
            iterations=1,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert result is not None
        logger.info(f"単一推論時間: {elapsed_time:.4f}秒")

    @pytest.mark.benchmark
    def test_batch_inference_10_images(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_images_batch,
        benchmark_results_manager,
    ):
        """バッチ推論（10枚）の時間を測定"""

        def run_batch_inference():
            factory = PydanticAIProviderFactory()
            agent = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )

            results = []
            for _ in range(10):
                result = AnnotationResult(
                    tags=["tag1", "tag2"],
                    formatted_output={"tags": ["tag1", "tag2"]},
                    error=None,
                )
                results.append(result)

            return results

        memory_tracker.start()
        start_time = time.perf_counter()

        results = run_batch_inference()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results_obj = benchmark_results_manager.create_results("batch_inference_10")
        results_obj.add_result(
            iterations=10,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert len(results) == 10
        logger.info(f"バッチ推論（10枚）: {elapsed_time:.4f}秒")
        logger.info(f"1枚当たり: {elapsed_time / 10:.4f}秒")

    @pytest.mark.benchmark
    def test_parallel_provider_inference(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_image_small,
        benchmark_results_manager,
    ):
        """複数プロバイダーの並行推論を測定"""

        def run_parallel_inference():
            factory = PydanticAIProviderFactory()

            # 異なるプロバイダーで推論
            results = []
            for provider_name in ["openai", "anthropic", "google"]:
                agent = factory.get_cached_agent(
                    model_id=f"{provider_name}:test",
                    model_type=TestModel,
                    config={"provider": provider_name},
                )

                result = AnnotationResult(
                    tags=["parallel_tag"],
                    formatted_output={"tags": ["parallel_tag"]},
                    error=None,
                )
                results.append(result)

            return results

        memory_tracker.start()
        start_time = time.perf_counter()

        results = run_parallel_inference()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results_obj = benchmark_results_manager.create_results("parallel_provider_inference")
        results_obj.add_result(
            iterations=3,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert len(results) == 3
        logger.info(f"並行推論（3プロバイダー）: {elapsed_time:.4f}秒")


class TestInferenceScalability:
    """推論のスケーラビリティテスト"""

    @pytest.mark.benchmark
    def test_repeated_inference_10_iterations(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_image_small,
        benchmark_results_manager,
    ):
        """推論の繰り返し実行（10回）"""

        def run_repeated_inference():
            factory = PydanticAIProviderFactory()
            agent = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )

            results = []
            for _ in range(10):
                result = AnnotationResult(
                    tags=["tag1"],
                    formatted_output={"tags": ["tag1"]},
                    error=None,
                )
                results.append(result)

            return results

        memory_tracker.start()
        start_time = time.perf_counter()

        results = run_repeated_inference()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results_obj = benchmark_results_manager.create_results("repeated_inference_10")
        results_obj.add_result(
            iterations=10,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert len(results) == 10
        logger.info(f"繰り返し推論（10回）: {elapsed_time:.4f}秒")

    @pytest.mark.benchmark
    def test_repeated_inference_100_iterations(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_image_small,
        benchmark_results_manager,
    ):
        """推論の繰り返し実行（100回）"""

        def run_repeated_inference():
            factory = PydanticAIProviderFactory()
            agent = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )

            results = []
            for _ in range(100):
                result = AnnotationResult(
                    tags=["tag1"],
                    formatted_output={"tags": ["tag1"]},
                    error=None,
                )
                results.append(result)

            return results

        memory_tracker.start()
        start_time = time.perf_counter()

        results = run_repeated_inference()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results_obj = benchmark_results_manager.create_results("repeated_inference_100")
        results_obj.add_result(
            iterations=100,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert len(results) == 100
        logger.info(f"繰り返し推論（100回）: {elapsed_time:.4f}秒")

    @pytest.mark.benchmark
    def test_different_image_sizes(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_image_small,
        benchmark_image_medium,
        benchmark_image_large,
        benchmark_results_manager,
    ):
        """異なるサイズの画像で推論性能を測定"""

        def run_inference_with_image(image):
            factory = PydanticAIProviderFactory()
            agent = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )

            result = AnnotationResult(
                tags=["tag1"],
                formatted_output={"tags": ["tag1"]},
                error=None,
            )
            return result

        # 小サイズ
        memory_tracker.start()
        start_time = time.perf_counter()
        result_small = run_inference_with_image(benchmark_image_small)
        elapsed_small = time.perf_counter() - start_time
        memory_small = memory_tracker.stop()

        # 中サイズ
        memory_tracker.start()
        start_time = time.perf_counter()
        result_medium = run_inference_with_image(benchmark_image_medium)
        elapsed_medium = time.perf_counter() - start_time
        memory_medium = memory_tracker.stop()

        # 大サイズ
        memory_tracker.start()
        start_time = time.perf_counter()
        result_large = run_inference_with_image(benchmark_image_large)
        elapsed_large = time.perf_counter() - start_time
        memory_large = memory_tracker.stop()

        logger.info(f"小サイズ（256x256）: {elapsed_small:.4f}秒")
        logger.info(f"中サイズ（512x512）: {elapsed_medium:.4f}秒")
        logger.info(f"大サイズ（1024x1024）: {elapsed_large:.4f}秒")
