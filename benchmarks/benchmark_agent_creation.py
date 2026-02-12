"""
Agent 生成時間ベンチマーク

PydanticAI Agent の初期化・キャッシング性能を測定。
"""

import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory

logger = logging.getLogger(__name__)


class TestAgentCreationBenchmark:
    """Agent 生成性能ベンチマーク"""

    @pytest.mark.benchmark
    def test_first_agent_creation_time(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """初回 Agent 生成時間を測定

        Agent インスタンスの初期化オーバーヘッドを計測。
        """

        def create_first_agent():
            factory = PydanticAIProviderFactory()
            # TestModel でプロバイダーを初期化
            agent = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )
            return agent

        memory_tracker.start()
        start_time = time.perf_counter()

        agent = create_first_agent()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("first_agent_creation")
        results.add_result(
            iterations=1,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert agent is not None
        logger.info(f"初回Agent生成時間: {elapsed_time:.4f}秒")
        logger.info(f"メモリ使用量: {memory_info['allocated_mb']:.2f}MB")

    @pytest.mark.benchmark
    def test_agent_cache_hit_time(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_config,
        benchmark_results_manager,
    ):
        """Agent キャッシュヒット時の時間を測定

        同じモデル設定で複数回取得した場合の高速化を確認。
        """
        factory = PydanticAIProviderFactory()

        # 初回作成
        agent_1 = factory.get_cached_agent(
            model_id="test",
            model_type=TestModel,
            config={},
        )

        # キャッシュヒット測定
        memory_tracker.start()
        start_time = time.perf_counter()

        for _ in range(benchmark_config["medium_iterations"]):
            agent_cached = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("agent_cache_hit")
        results.add_result(
            iterations=benchmark_config["medium_iterations"],
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        # キャッシュが有効なことを確認
        assert agent_cached is agent_1
        logger.info(f"キャッシュヒット（100回）: {elapsed_time:.4f}秒")
        logger.info(f"1回当たり: {elapsed_time / benchmark_config['medium_iterations']:.6f}秒")

    @pytest.mark.benchmark
    def test_multiple_model_cache_overhead(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_config,
        benchmark_results_manager,
    ):
        """複数モデルでのキャッシュオーバーヘッド測定

        異なるモデルIDでの Agent 生成オーバーヘッド。
        """
        factory = PydanticAIProviderFactory()
        model_ids = [f"test_model_{i}" for i in range(10)]

        memory_tracker.start()
        start_time = time.perf_counter()

        agents = []
        for model_id in model_ids:
            agent = factory.get_cached_agent(
                model_id=model_id,
                model_type=TestModel,
                config={},
            )
            agents.append(agent)

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("multiple_model_cache")
        results.add_result(
            iterations=len(model_ids),
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert len(agents) == len(model_ids)
        logger.info(f"複数モデル（10個）キャッシュ: {elapsed_time:.4f}秒")
        logger.info(f"1モデル当たり: {elapsed_time / len(model_ids):.4f}秒")

    @pytest.mark.benchmark
    def test_provider_instance_reuse(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_config,
        benchmark_results_manager,
    ):
        """プロバイダーインスタンス再利用性能を測定

        同じプロバイダーで異なるモデルを使用した場合の効率。
        """
        factory = PydanticAIProviderFactory()

        # 初回：プロバイダー生成
        memory_tracker.start()
        start_time = time.perf_counter()

        provider_1 = factory.get_cached_agent(
            model_id="openai:model1",
            model_type=TestModel,
            config={"provider": "openai"},
        )

        elapsed_time_first = time.perf_counter() - start_time
        memory_info_first = memory_tracker.stop()

        # 2回目：同じプロバイダー
        memory_tracker.start()
        start_time = time.perf_counter()

        provider_2 = factory.get_cached_agent(
            model_id="openai:model2",
            model_type=TestModel,
            config={"provider": "openai"},
        )

        elapsed_time_second = time.perf_counter() - start_time
        memory_info_second = memory_tracker.stop()

        results_first = benchmark_results_manager.create_results("provider_reuse_first")
        results_first.add_result(
            iterations=1,
            elapsed_time=elapsed_time_first,
            memory_info=memory_info_first,
        )

        results_second = benchmark_results_manager.create_results("provider_reuse_second")
        results_second.add_result(
            iterations=1,
            elapsed_time=elapsed_time_second,
            memory_info=memory_info_second,
        )

        logger.info(f"初回プロバイダー生成: {elapsed_time_first:.4f}秒")
        logger.info(f"プロバイダー再利用: {elapsed_time_second:.4f}秒")
        logger.info(f"高速化率: {elapsed_time_first / elapsed_time_second:.1f}倍")


class TestAgentCreationScalability:
    """Agent 生成のスケーラビリティテスト"""

    @pytest.mark.benchmark
    def test_cache_scaling_10_agents(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """10個の Agent キャッシング"""

        def create_agents():
            factory = PydanticAIProviderFactory()
            return [
                factory.get_cached_agent(
                    model_id=f"test_{i}",
                    model_type=TestModel,
                    config={},
                )
                for i in range(10)
            ]

        memory_tracker.start()
        start_time = time.perf_counter()

        agents = create_agents()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("cache_scaling_10")
        results.add_result(
            iterations=10,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert len(agents) == 10
        logger.info(f"10 Agent キャッシング: {elapsed_time:.4f}秒")

    @pytest.mark.benchmark
    def test_cache_scaling_100_agents(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """100個の Agent キャッシング"""

        def create_agents():
            factory = PydanticAIProviderFactory()
            return [
                factory.get_cached_agent(
                    model_id=f"test_{i}",
                    model_type=TestModel,
                    config={},
                )
                for i in range(100)
            ]

        memory_tracker.start()
        start_time = time.perf_counter()

        agents = create_agents()

        elapsed_time = time.perf_counter() - start_time
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("cache_scaling_100")
        results.add_result(
            iterations=100,
            elapsed_time=elapsed_time,
            memory_info=memory_info,
        )

        assert len(agents) == 100
        logger.info(f"100 Agent キャッシング: {elapsed_time:.4f}秒")
