"""
メモリ使用量ベンチマーク

PydanticAI Agent とプロバイダーのメモリ効率を測定。
"""

import logging
import time

import pytest
from pydantic_ai.models.test import TestModel

from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory

logger = logging.getLogger(__name__)


class TestMemoryBenchmark:
    """メモリ使用量ベンチマーク"""

    @pytest.mark.benchmark
    def test_single_agent_memory(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """単一 Agent のメモリ使用量を測定"""

        def create_agent():
            factory = PydanticAIProviderFactory()
            agent = factory.get_cached_agent(
                model_id="test",
                model_type=TestModel,
                config={},
            )
            return agent

        memory_tracker.start()
        agent = create_agent()
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("single_agent_memory")
        results.add_result(
            iterations=1,
            elapsed_time=0.0,
            memory_info=memory_info,
        )

        assert agent is not None
        logger.info(
            f"単一 Agent メモリ使用量: "
            f"現在={memory_info['current_memory_mb']:.2f}MB, "
            f"ピーク={memory_info['peak_memory_mb']:.2f}MB"
        )

    @pytest.mark.benchmark
    def test_multiple_agents_memory(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """複数 Agent のメモリ使用量を測定"""

        def create_agents(count):
            factory = PydanticAIProviderFactory()
            agents = []
            for i in range(count):
                agent = factory.get_cached_agent(
                    model_id=f"test_{i}",
                    model_type=TestModel,
                    config={},
                )
                agents.append(agent)
            return agents

        memory_tracker.start()
        agents = create_agents(10)
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("multiple_agents_memory")
        results.add_result(
            iterations=10,
            elapsed_time=0.0,
            memory_info=memory_info,
        )

        assert len(agents) == 10
        logger.info(
            f"複数 Agent（10個）メモリ使用量: "
            f"確保={memory_info['allocated_mb']:.2f}MB, "
            f"ピーク={memory_info['peak_memory_mb']:.2f}MB, "
            f"1Agent当たり={memory_info['allocated_mb'] / 10:.2f}MB"
        )

    @pytest.mark.benchmark
    def test_provider_sharing_memory_efficiency(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """プロバイダー共有によるメモリ効率を測定

        同じプロバイダーで複数モデルを使用した場合のメモリ効率。
        """
        factory = PydanticAIProviderFactory()

        # 初回：プロバイダー生成
        memory_tracker.start()
        agent_1 = factory.get_cached_agent(
            model_id="openai:model1",
            model_type=TestModel,
            config={"provider": "openai"},
        )
        memory_info_1 = memory_tracker.stop()

        # 2回目以降：同じプロバイダー
        memory_tracker.start()
        agent_2 = factory.get_cached_agent(
            model_id="openai:model2",
            model_type=TestModel,
            config={"provider": "openai"},
        )
        memory_info_2 = memory_tracker.stop()

        memory_tracker.start()
        agent_3 = factory.get_cached_agent(
            model_id="openai:model3",
            model_type=TestModel,
            config={"provider": "openai"},
        )
        memory_info_3 = memory_tracker.stop()

        results_1 = benchmark_results_manager.create_results("provider_sharing_1")
        results_1.add_result(
            iterations=1,
            elapsed_time=0.0,
            memory_info=memory_info_1,
        )

        results_2 = benchmark_results_manager.create_results("provider_sharing_2")
        results_2.add_result(
            iterations=1,
            elapsed_time=0.0,
            memory_info=memory_info_2,
        )

        results_3 = benchmark_results_manager.create_results("provider_sharing_3")
        results_3.add_result(
            iterations=1,
            elapsed_time=0.0,
            memory_info=memory_info_3,
        )

        logger.info(
            f"プロバイダー共有メモリ効率:"
            f"\n  初回（model1）: {memory_info_1['allocated_mb']:.2f}MB"
            f"\n  2番目（model2）: {memory_info_2['allocated_mb']:.2f}MB"
            f"\n  3番目（model3）: {memory_info_3['allocated_mb']:.2f}MB"
        )

        # 共有効果：2回目以降はメモリ増加が少ないはず
        assert memory_info_2["allocated_mb"] < memory_info_1["allocated_mb"]
        assert memory_info_3["allocated_mb"] < memory_info_1["allocated_mb"]

    @pytest.mark.benchmark
    def test_cache_memory_scaling(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """キャッシュのメモリスケーリングを測定

        Agent キャッシュサイズの増加に伴うメモリ使用量。
        """
        factory = PydanticAIProviderFactory()

        results_list = []

        for cache_size in [5, 10, 25, 50]:
            memory_tracker.start()

            agents = []
            for i in range(cache_size):
                agent = factory.get_cached_agent(
                    model_id=f"test_{i}",
                    model_type=TestModel,
                    config={},
                )
                agents.append(agent)

            memory_info = memory_tracker.stop()

            results = benchmark_results_manager.create_results(f"cache_size_{cache_size}")
            results.add_result(
                iterations=cache_size,
                elapsed_time=0.0,
                memory_info=memory_info,
            )

            results_list.append(
                {
                    "cache_size": cache_size,
                    "allocated_mb": memory_info["allocated_mb"],
                    "peak_mb": memory_info["peak_memory_mb"],
                }
            )

        logger.info("キャッシュサイズ別メモリ使用量:")
        for result in results_list:
            logger.info(
                f"  {result['cache_size']}個: "
                f"確保={result['allocated_mb']:.2f}MB, "
                f"ピーク={result['peak_mb']:.2f}MB"
            )

    @pytest.mark.benchmark
    def test_agent_cleanup_memory_release(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """Agent クリーンアップ後のメモリ解放を測定"""
        import gc

        factory = PydanticAIProviderFactory()

        # 作成
        memory_tracker.start()
        agents = []
        for i in range(10):
            agent = factory.get_cached_agent(
                model_id=f"test_{i}",
                model_type=TestModel,
                config={},
            )
            agents.append(agent)
        memory_after_creation = memory_tracker.stop()

        # クリーンアップ
        agents.clear()
        gc.collect()

        memory_tracker.start()
        # メモリ状態を確認
        memory_after_cleanup = memory_tracker.stop()

        results = benchmark_results_manager.create_results("agent_cleanup_memory_release")
        results.add_result(
            iterations=10,
            elapsed_time=0.0,
            memory_info=memory_after_cleanup,
        )

        logger.info(
            f"メモリ解放:"
            f"\n  作成後: {memory_after_creation['peak_memory_mb']:.2f}MB"
            f"\n  クリーンアップ後: {memory_after_cleanup['peak_memory_mb']:.2f}MB"
        )


class TestMemoryScalability:
    """メモリのスケーラビリティテスト"""

    @pytest.mark.benchmark
    def test_100_agents_memory(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """100個の Agent によるメモリ使用量"""

        def create_many_agents():
            factory = PydanticAIProviderFactory()
            agents = []
            for i in range(100):
                agent = factory.get_cached_agent(
                    model_id=f"test_{i}",
                    model_type=TestModel,
                    config={},
                )
                agents.append(agent)
            return agents

        memory_tracker.start()
        agents = create_many_agents()
        memory_info = memory_tracker.stop()

        results = benchmark_results_manager.create_results("100_agents_memory")
        results.add_result(
            iterations=100,
            elapsed_time=0.0,
            memory_info=memory_info,
        )

        assert len(agents) == 100
        logger.info(
            f"100 Agent メモリ使用量:"
            f"\n  確保: {memory_info['allocated_mb']:.2f}MB"
            f"\n  ピーク: {memory_info['peak_memory_mb']:.2f}MB"
            f"\n  1Agent当たり: {memory_info['allocated_mb'] / 100:.2f}MB"
        )

    @pytest.mark.benchmark
    def test_provider_memory_per_type(
        self,
        benchmark,
        mock_api_key,
        memory_tracker,
        benchmark_results_manager,
    ):
        """プロバイダータイプ別のメモリ使用量"""
        factory = PydanticAIProviderFactory()

        provider_results = {}

        for provider_name in ["openai", "anthropic", "google", "openrouter"]:
            memory_tracker.start()

            agents = []
            for i in range(5):
                agent = factory.get_cached_agent(
                    model_id=f"{provider_name}:model_{i}",
                    model_type=TestModel,
                    config={"provider": provider_name},
                )
                agents.append(agent)

            memory_info = memory_tracker.stop()

            provider_results[provider_name] = {
                "allocated_mb": memory_info["allocated_mb"],
                "per_agent_mb": memory_info["allocated_mb"] / 5,
            }

            results = benchmark_results_manager.create_results(f"provider_memory_{provider_name}")
            results.add_result(
                iterations=5,
                elapsed_time=0.0,
                memory_info=memory_info,
            )

        logger.info("プロバイダー別メモリ使用量:")
        for provider_name, info in provider_results.items():
            logger.info(
                f"  {provider_name}: "
                f"確保={info['allocated_mb']:.2f}MB, "
                f"1Agent当たり={info['per_agent_mb']:.2f}MB"
            )
