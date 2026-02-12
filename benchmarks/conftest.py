"""
ベンチマークテスト用共通フィクスチャとユーティリティ

PydanticAI Model Factory の性能ベンチマークに必要な
共通フィクスチャ、メモリ計測、メトリクス管理を提供。
"""

import gc
import logging
import os
import tracemalloc
from typing import Any, Callable

import pytest
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# テスト画像フィクスチャ
# ============================================================================


@pytest.fixture
def benchmark_image_small():
    """ベンチマーク用の小さな画像（256x256）"""
    return Image.new("RGB", (256, 256), color="red")


@pytest.fixture
def benchmark_image_medium():
    """ベンチマーク用の中サイズ画像（512x512）"""
    return Image.new("RGB", (512, 512), color="blue")


@pytest.fixture
def benchmark_image_large():
    """ベンチマーク用の大きな画像（1024x1024）"""
    return Image.new("RGB", (1024, 1024), color="green")


@pytest.fixture
def benchmark_images_batch(benchmark_image_small):
    """ベンチマーク用の画像バッチ（10枚）"""
    return [benchmark_image_small for _ in range(10)]


# ============================================================================
# メモリ計測ユーティリティ
# ============================================================================


class MemoryTracker:
    """メモリ使用量を計測するユーティリティ"""

    def __init__(self):
        """初期化"""
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0

    def start(self):
        """メモリ計測開始"""
        gc.collect()
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]

    def stop(self) -> dict[str, float]:
        """メモリ計測終了と結果を返す

        Returns:
            メモリ使用量情報（単位: MB）:
            - start_memory: 開始時のメモリ
            - peak_memory: ピークメモリ
            - current_memory: 終了時のメモリ
            - allocated: 確保されたメモリ
        """
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "start_memory_mb": self.start_memory / 1024 / 1024,
            "current_memory_mb": current / 1024 / 1024,
            "peak_memory_mb": peak / 1024 / 1024,
            "allocated_mb": (current - self.start_memory) / 1024 / 1024,
        }


@pytest.fixture
def memory_tracker():
    """メモリ計測用フィクスチャ"""
    return MemoryTracker()


# ============================================================================
# ベンチマーク設定フィクスチャ
# ============================================================================


@pytest.fixture
def benchmark_config():
    """ベンチマーク設定"""
    return {
        "small_iterations": 10,
        "medium_iterations": 100,
        "large_iterations": 1000,
        "warmup_iterations": 1,
        "timeout": 300,  # 5分
    }


@pytest.fixture
def mock_api_key(monkeypatch):
    """テスト用モック API キー設定"""
    test_keys = {
        "OPENAI_API_KEY": "sk-test-key-" + "x" * 40,
        "ANTHROPIC_API_KEY": "sk-ant-test-key-" + "x" * 40,
        "GOOGLE_API_KEY": "test-google-key-" + "x" * 40,
        "OPENROUTER_API_KEY": "sk-or-test-key-" + "x" * 40,
    }

    for key, value in test_keys.items():
        monkeypatch.setenv(key, value)

    logger.info("モック API キーを設定しました")
    return test_keys


# ============================================================================
# ベンチマーク結果ユーティリティ
# ============================================================================


class BenchmarkResults:
    """ベンチマーク結果を管理するクラス"""

    def __init__(self, test_name: str):
        """初期化

        Args:
            test_name: テスト名
        """
        self.test_name = test_name
        self.results = []
        self.metrics = {}

    def add_result(self, iterations: int, elapsed_time: float, memory_info: dict[str, float]):
        """結果を追加

        Args:
            iterations: 反復回数
            elapsed_time: 経過時間（秒）
            memory_info: メモリ情報
        """
        result = {
            "iterations": iterations,
            "elapsed_time": elapsed_time,
            "time_per_iteration": elapsed_time / iterations if iterations > 0 else 0,
            "iterations_per_second": iterations / elapsed_time if elapsed_time > 0 else 0,
            "memory_info": memory_info,
        }
        self.results.append(result)

    def calculate_metrics(self):
        """平均メトリクスを計算"""
        if not self.results:
            return {}

        times_per_iteration = [r["time_per_iteration"] for r in self.results]
        memory_allocated = [r["memory_info"]["allocated_mb"] for r in self.results]

        self.metrics = {
            "avg_time_per_iteration": sum(times_per_iteration) / len(times_per_iteration),
            "min_time_per_iteration": min(times_per_iteration),
            "max_time_per_iteration": max(times_per_iteration),
            "avg_memory_allocated_mb": sum(memory_allocated) / len(memory_allocated),
            "total_peak_memory_mb": max(r["memory_info"]["peak_memory_mb"] for r in self.results),
        }
        return self.metrics

    def get_summary(self) -> str:
        """結果サマリーを取得

        Returns:
            フォーマットされたサマリーテキスト
        """
        metrics = self.calculate_metrics()

        if not metrics:
            return f"Test: {self.test_name} - No results"

        summary = f"""
ベンチマーク結果: {self.test_name}
{"=" * 50}
反復数: {len(self.results)}
平均時間/反復: {metrics["avg_time_per_iteration"]:.6f}秒
最小時間/反復: {metrics["min_time_per_iteration"]:.6f}秒
最大時間/反復: {metrics["max_time_per_iteration"]:.6f}秒
平均メモリ確保: {metrics["avg_memory_allocated_mb"]:.2f}MB
ピークメモリ: {metrics["total_peak_memory_mb"]:.2f}MB
"""
        return summary


@pytest.fixture
def benchmark_results_manager():
    """ベンチマーク結果管理用フィクスチャ"""

    class ResultsManager:
        def __init__(self):
            self.test_results = {}

        def create_results(self, test_name: str) -> BenchmarkResults:
            """新しい結果オブジェクトを作成"""
            results = BenchmarkResults(test_name)
            self.test_results[test_name] = results
            return results

        def get_all_results(self) -> dict[str, BenchmarkResults]:
            """すべての結果を取得"""
            return self.test_results

        def print_summary(self):
            """すべてのテスト結果を出力"""
            for test_name, results in self.test_results.items():
                logger.info(results.get_summary())

    return ResultsManager()


# ============================================================================
# パフォーマンス計測デコレータ
# ============================================================================


def measure_performance(iterations: int = 1):
    """パフォーマンス計測デコレータ

    Args:
        iterations: 反復回数
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            memory_tracker = MemoryTracker()
            memory_tracker.start()

            import time

            start_time = time.perf_counter()
            results = []

            for _ in range(iterations):
                result = func(*args, **kwargs)
                results.append(result)

            elapsed_time = time.perf_counter() - start_time
            memory_info = memory_tracker.stop()

            return {
                "results": results,
                "elapsed_time": elapsed_time,
                "iterations": iterations,
                "time_per_iteration": elapsed_time / iterations,
                "memory_info": memory_info,
            }

        return wrapper

    return decorator
