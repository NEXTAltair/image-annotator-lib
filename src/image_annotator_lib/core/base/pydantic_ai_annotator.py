"""PydanticAI統一WebAPIアノテーター - 最新仕様完全準拠

このモジュールは、PydanticAIの最新機能を最大限活用した統一WebAPIアノテーターを提供します。
従来のプロバイダー固有実装を統合し、PydanticAIのAgent機能を中心とした設計により、
すべてのWebAPIプロバイダー（Google, OpenAI, Anthropic, OpenRouter等）を
単一の実装で効率的に処理します。
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, ClassVar, Self

from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models import KnownModelName
from pydantic_ai.settings import ModelSettings

from ...exceptions.errors import ConfigurationError
from ..config import config_registry
from ..types import AnnotationResult, AnnotationSchema
from ..utils import logger
from .annotator import BaseAnnotator


@dataclass
class AnnotationAgentConfig:
    """画像アノテーション特化のPydanticAI Agent設定"""

    # 基本設定
    model_id: KnownModelName | str
    name: str

    # PydanticAI ModelSettings 完全活用
    model_settings: ModelSettings = field(
        default_factory=lambda: ModelSettings(
            temperature=0.1,  # アノテーションは一貫性重視
            max_tokens=1800,  # 詳細説明対応
            timeout=30.0,  # レスポンス制御
            top_p=0.9,  # 適度な多様性
        )
    )

    # 高度機能
    retries: int = 3
    output_retries: int = 2
    enable_streaming: bool = False
    batch_size: int = 1

    # 監視・テレメトリ
    instrument: bool = True  # OpenTelemetry監視

    # システムプロンプト設定
    focus_quality: bool = False
    analyze_style: bool = False
    custom_instructions: str = ""


class AdvancedAgentFactory:
    """PydanticAI Agent高度管理ファクトリ"""

    _agent_cache: ClassVar[dict[str, Agent]] = {}
    _config_hashes: ClassVar[dict[str, str]] = {}

    @classmethod
    def create_optimized_agent(cls, config: AnnotationAgentConfig) -> Agent[None, AnnotationSchema]:
        """最適化されたAgent作成・キャッシング"""

        config_hash = cls._calculate_config_hash(config)
        cache_key = f"{config.model_id}:{config_hash}"

        # 設定変更チェック
        if cache_key in cls._agent_cache and cls._config_hashes.get(cache_key) == config_hash:
            logger.debug(f"Agent cache hit: {cache_key}")
            return cls._agent_cache[cache_key]

        # 新規Agent作成
        logger.debug(f"Creating new optimized Agent: {cache_key}")

        try:
            # PydanticAI最新機能フル活用
            agent = Agent(
                model=config.model_id,  # infer_model()自動呼び出し
                output_type=AnnotationSchema,
                name=config.name,
                # システムプロンプト階層化
                system_prompt=cls._build_system_prompts(config),
                instructions=cls._build_instructions(config),
                # ModelSettings完全活用
                model_settings=config.model_settings,
                # 高度機能
                retries=config.retries,
                output_retries=config.output_retries,
                instrument=config.instrument,
                # 将来拡張用
                tools=[],  # 必要に応じてツール追加
                history_processors=[],  # 会話履歴処理
            )

            # キャッシュ更新
            cls._agent_cache[cache_key] = agent
            cls._config_hashes[cache_key] = config_hash

            return agent

        except Exception as e:
            logger.error(f"Agent creation failed for {config.model_id}: {e}")
            raise ConfigurationError(f"Failed to create PydanticAI Agent: {e}") from e

    @classmethod
    def _calculate_config_hash(cls, config: AnnotationAgentConfig) -> str:
        """設定のハッシュ値計算 - キャッシュ無効化判定用"""
        config_str = json.dumps(
            {
                "model_id": config.model_id,
                "model_settings": dict(config.model_settings),
                "retries": config.retries,
                "output_retries": config.output_retries,
                "focus_quality": config.focus_quality,
                "analyze_style": config.analyze_style,
                "custom_instructions": config.custom_instructions,
            },
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    @classmethod
    def _build_system_prompts(cls, config: AnnotationAgentConfig) -> Sequence[str]:
        """動的システムプロンプト構築"""
        prompts = [
            "You are an expert image annotation AI specialized in analyzing visual content.",
            f"Your name is {config.name}.",
            "Analyze images with precision and provide structured annotations in the specified JSON schema format.",
        ]

        # モデル特化プロンプト追加
        model_id_lower = config.model_id.lower()
        if "gpt" in model_id_lower or "openai" in model_id_lower:
            prompts.append("Leverage your multimodal capabilities for detailed visual analysis.")
        elif "claude" in model_id_lower or "anthropic" in model_id_lower:
            prompts.append("Use your advanced reasoning for comprehensive image understanding.")
        elif "gemini" in model_id_lower or "google" in model_id_lower:
            prompts.append("Apply your visual intelligence for accurate scene interpretation.")

        # 特化機能プロンプト
        if config.focus_quality:
            prompts.append("Pay special attention to image quality and aesthetic elements.")

        if config.analyze_style:
            prompts.append("Include analysis of artistic style and visual techniques.")

        return prompts

    @classmethod
    def _build_instructions(cls, config: AnnotationAgentConfig) -> str:
        """動的instruction構築"""
        instructions = [
            "Analyze the provided image carefully and comprehensively.",
            "Focus on accuracy and detail in your annotations.",
            "Use clear, descriptive language in captions.",
            "Generate relevant tags that capture key visual elements.",
            "Return results in the exact JSON schema format specified.",
        ]

        # カスタム指示追加
        if config.custom_instructions:
            instructions.append(config.custom_instructions)

        return " ".join(instructions)

    @classmethod
    def clear_cache(cls) -> None:
        """キャッシュクリア - テスト用"""
        cls._agent_cache.clear()
        cls._config_hashes.clear()
        logger.debug("Agent cache cleared")


class PydanticAIWebAPIAnnotator(BaseAnnotator):
    """PydanticAI全機能活用の統一WebAPIアノテーター

    このクラスは、PydanticAIの最新機能を活用して全WebAPIプロバイダーを統一的に処理します。
    従来の複雑なプロバイダー固有実装を削除し、PydanticAIのAgent中心設計により
    シンプルで効率的な実装を実現しています。

    主な特徴:
    - すべてのWebAPIプロバイダー（Google, OpenAI, Anthropic, OpenRouter）を統一処理
    - PydanticAI infer_model()による自動プロバイダー判定・最適化
    - Agent-levelキャッシングによる高性能化
    - OpenTelemetry統合監視
    - ストリーミング・バッチ処理対応
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.config = self._build_agent_config()
        self.agent: Agent[None, AnnotationSchema] | None = None

    def _build_agent_config(self) -> AnnotationAgentConfig:
        """設定からPydanticAI最適化設定を構築"""
        api_model_id = config_registry.get(self.model_name, "api_model_id")
        if not api_model_id:
            raise ConfigurationError(f"Model {self.model_name} missing required api_model_id")

        # ModelSettings動的構築
        model_settings = ModelSettings()

        # 設定から動的にModelSettingsを構築
        model_settings_dict = {}
        if temp := config_registry.get(self.model_name, "temperature", default=None):
            model_settings_dict["temperature"] = float(temp)
        if max_tokens := config_registry.get(self.model_name, "max_tokens", default=None):
            model_settings_dict["max_tokens"] = int(max_tokens)
        if timeout := config_registry.get(self.model_name, "timeout", default=None):
            model_settings_dict["timeout"] = float(timeout)
        if top_p := config_registry.get(self.model_name, "top_p", default=None):
            model_settings_dict["top_p"] = float(top_p)
        if seed := config_registry.get(self.model_name, "seed", default=None):
            model_settings_dict["seed"] = int(seed)

        # ModelSettingsを構築
        model_settings = ModelSettings(**model_settings_dict) if model_settings_dict else model_settings

        return AnnotationAgentConfig(
            model_id=api_model_id,
            name=f"ImageAnnotator-{self.model_name}",
            model_settings=model_settings,
            retries=config_registry.get(self.model_name, "retries", 3),
            output_retries=config_registry.get(self.model_name, "output_retries", 2),
            enable_streaming=config_registry.get(self.model_name, "enable_streaming", False),
            batch_size=config_registry.get(self.model_name, "batch_size", 1),
            instrument=config_registry.get(self.model_name, "instrument", True),
            focus_quality=config_registry.get(self.model_name, "focus_quality", False),
            analyze_style=config_registry.get(self.model_name, "analyze_style", False),
            custom_instructions=config_registry.get(self.model_name, "custom_instructions", ""),
        )

    def __enter__(self) -> Self:
        """PydanticAI Agent統一初期化"""
        try:
            logger.info(f"Initializing PydanticAI Agent for model: {self.model_name}")

            # AdvancedAgentFactoryでAgent取得
            self.agent = AdvancedAgentFactory.create_optimized_agent(self.config)

            logger.info(
                f"PydanticAI Agent ready - Model: {self.config.model_id}, "
                f"Settings: {dict(self.config.model_settings)}"
            )

            return self

        except Exception as e:
            logger.error(
                f"Agent initialization failed for {self.model_name}",
                extra={
                    "model_name": self.model_name,
                    "model_id": self.config.model_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise ConfigurationError(f"Failed to initialize PydanticAI Agent: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """統一クリーンアップ"""
        if self.agent:
            logger.debug(f"Releasing PydanticAI Agent: {self.model_name}")
            # PydanticAI Agentは自動管理されるため明示的解放不要
            self.agent = None

    def predict(
        self, images: list[Image.Image], phash_list: list[str] | None = None
    ) -> list[AnnotationResult]:
        """統一予測インターフェース - 同期ラッパー"""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Use within context manager.")

        # 非同期実行を同期ラッパーで実行
        loop = asyncio.get_event_loop()
        try:
            return loop.run_until_complete(self.predict_async(images, phash_list))
        except RuntimeError:
            # 新しいイベントループを作成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.predict_async(images, phash_list))
            finally:
                loop.close()

    async def predict_async(
        self, images: list[Image.Image], phash_list: list[str] | None = None
    ) -> list[AnnotationResult]:
        """完全非同期実装"""
        if not self.agent:
            raise RuntimeError("Agent not initialized")

        if self.config.enable_streaming:
            return await self._predict_streaming(images, phash_list)
        else:
            return await self._predict_batch(images, phash_list)

    async def _predict_streaming(
        self, images: list[Image.Image], phash_list: list[str] | None = None
    ) -> list[AnnotationResult]:
        """ストリーミング処理 - リアルタイム結果"""
        results = []

        for i, image in enumerate(images):
            phash = phash_list[i] if phash_list and i < len(phash_list) else None

            try:
                binary_content = self._image_to_binary_content(image)

                # PydanticAI Streaming API
                async with self.agent.run_stream(message_history=[binary_content]) as stream:
                    # ストリーミング結果の集約
                    accumulated_response = None
                    async for chunk in stream:
                        if hasattr(chunk, "event_type") and chunk.event_type == "output":
                            accumulated_response = chunk.data

                if accumulated_response:
                    results.append(
                        AnnotationResult(
                            phash=phash,
                            tags=accumulated_response.tags,
                            formatted_output=accumulated_response,
                            error=None,
                        )
                    )
                else:
                    results.append(
                        AnnotationResult(
                            phash=phash, tags=[], formatted_output=None, error="No response from streaming"
                        )
                    )

            except Exception as e:
                logger.error(f"Streaming prediction failed for {self.model_name}: {e}")
                results.append(AnnotationResult(phash=phash, tags=[], formatted_output=None, error=str(e)))

        return results

    async def _predict_batch(
        self, images: list[Image.Image], phash_list: list[str] | None = None
    ) -> list[AnnotationResult]:
        """バッチ処理 - 効率性重視"""
        batch_size = self.config.batch_size
        results = []

        # バッチ分割処理
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_phashes = phash_list[i : i + batch_size] if phash_list else None

            # 並列処理タスク準備
            batch_tasks = []
            for j, image in enumerate(batch_images):
                phash = batch_phashes[j] if batch_phashes and j < len(batch_phashes) else None
                binary_content = self._image_to_binary_content(image)

                task = self.agent.run(message_history=[binary_content])
                batch_tasks.append((task, phash))

            # 並列実行
            batch_results = await asyncio.gather(*[task for task, _ in batch_tasks], return_exceptions=True)

            # 結果処理
            for result, (_, phash) in zip(batch_results, batch_tasks, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Batch prediction failed for {self.model_name}: {result}")
                    results.append(
                        AnnotationResult(phash=phash, tags=[], formatted_output=None, error=str(result))
                    )
                else:
                    results.append(
                        AnnotationResult(
                            phash=phash, tags=result.data.tags, formatted_output=result.data, error=None
                        )
                    )

        return results

    def _image_to_binary_content(self, image: Image.Image) -> BinaryContent:
        """PydanticAI標準の画像変換処理"""
        buffer = BytesIO()
        # WebP形式で効率的に圧縮
        image.save(buffer, format="WEBP", quality=85, optimize=True)
        return BinaryContent(data=buffer.getvalue(), media_type="image/webp")

    # BaseAnnotator抽象メソッド最小実装（PydanticAIが処理するため使用されない）
    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """PydanticAIで処理されるため実装不要"""
        return images

    def _run_inference(self, processed: Any) -> Any:
        """PydanticAIで処理されるため実装不要"""
        return processed

    def _format_predictions(self, raw_outputs: Any) -> Any:
        """PydanticAIで処理されるため実装不要"""
        return raw_outputs

    def _generate_tags(self, formatted_output: Any) -> list[str]:
        """PydanticAIで処理されるため実装不要"""
        if hasattr(formatted_output, "tags"):
            return formatted_output.tags
        elif isinstance(formatted_output, dict) and "tags" in formatted_output:
            return formatted_output["tags"]
        return []


__all__ = ["AdvancedAgentFactory", "AnnotationAgentConfig", "PydanticAIWebAPIAnnotator"]
