"""Simplified PydanticAI Agent factory with API discovery integration."""

from typing import Any

from pydantic_ai import Agent

from .api_model_discovery import discover_available_vision_models
from .simple_config import get_model_settings
from .types import AnnotationSchema
from .utils import logger


class SimplifiedAgentFactory:
    """Factory for creating PydanticAI Agents with simplified configuration."""

    def __init__(self) -> None:
        self._available_models: list[str] = []
        self._all_models_data: dict[str, Any] = {}
        self._agents_cache: dict[str, Agent] = {}

    def refresh_available_models(self, force_refresh: bool = False) -> list[str]:
        """
        Refresh the list of available models from API discovery.

        Args:
            force_refresh: If True, force refresh from API

        Returns:
            List of available model IDs
        """
        try:
            result = discover_available_vision_models(force_refresh=force_refresh)
            if "models" in result:
                # discovery の toml_data を直接使用する。
                # TOML 書き込み失敗時も in-memory の正確なデータが渡されるため
                # 別途 load_available_api_models() を呼ぶ必要がない。
                toml_data: dict[str, Any] = result.get("toml_data", {})
                self._all_models_data = toml_data
                self._available_models = [
                    mid for mid, data in toml_data.items()
                    if isinstance(data, dict) and data.get("deprecated_on") is None
                ]
                logger.info(
                    f"利用可能モデル: {len(self._available_models)} 件"
                    f" (全 {len(toml_data)} 件中)"
                )
            else:
                logger.error(f"Failed to discover models: {result.get('error', 'Unknown error')}")
                if not self._available_models:
                    self._available_models = []
        except Exception as e:
            logger.error(f"Error during model discovery: {e}")
            if not self._available_models:
                self._available_models = []

        return self._available_models

    def get_available_models(self) -> list[str]:
        """
        Get list of available models. Refresh if empty.

        Returns:
            List of available model IDs
        """
        if not self._available_models:
            self.refresh_available_models()
        return self._available_models

    def create_agent(self, model_id: str, **kwargs) -> Agent:
        """
        Create a PydanticAI Agent with simplified configuration.

        Args:
            model_id: The model ID (e.g., "google/gemini-2.5-pro-preview-03-25")
            **kwargs: Additional Agent parameters (override defaults)

        Returns:
            Configured PydanticAI Agent
        """
        # Get model settings from simplified config
        model_settings = get_model_settings(model_id)

        # Merge with any additional kwargs
        final_settings = model_settings.copy()
        final_settings.update(kwargs)

        # Remove non-Agent parameters
        agent_params = {
            k: v
            for k, v in final_settings.items()
            if k in ["max_output_tokens", "timeout", "temperature", "top_p"]
        }

        logger.debug(f"Creating Agent for {model_id} with settings: {agent_params}")

        # Create Agent with structured output schema
        agent = Agent(model=model_id, result_type=AnnotationSchema, **agent_params)

        return agent

    def get_cached_agent(self, model_id: str, **kwargs) -> Agent:
        """
        Get cached Agent or create new one.

        Args:
            model_id: The model ID
            **kwargs: Additional Agent parameters

        Returns:
            Cached or newly created Agent
        """
        cache_key = f"{model_id}:{hash(frozenset(kwargs.items()))}"

        if cache_key not in self._agents_cache:
            self._agents_cache[cache_key] = self.create_agent(model_id, **kwargs)

        return self._agents_cache[cache_key]

    def list_all_models(self) -> list[str]:
        """廃止済みモデルを含む全モデル ID のリストを返す。"""
        if not self._all_models_data:
            self.refresh_available_models()
        return list(self._all_models_data.keys())

    def is_model_deprecated(self, model_id: str) -> bool:
        """指定モデルが廃止済みかどうかを確認する。"""
        if not self._all_models_data:
            self.refresh_available_models()
        data = self._all_models_data.get(model_id)
        return isinstance(data, dict) and data.get("deprecated_on") is not None

    def is_model_available(self, model_id: str) -> bool:
        """
        Check if a model is available.

        Args:
            model_id: The model ID to check

        Returns:
            True if model is available
        """
        available_models = self.get_available_models()
        return model_id in available_models

    def get_models_by_provider(self, provider: str) -> list[str]:
        """
        Get models filtered by provider.

        Args:
            provider: Provider name (e.g., "google", "openai", "anthropic")

        Returns:
            List of model IDs for the specified provider
        """
        available_models = self.get_available_models()
        provider_models = [model for model in available_models if model.startswith(f"{provider}/")]
        return provider_models

    def clear_cache(self) -> None:
        """Clear the Agent cache."""
        self._agents_cache.clear()
        logger.debug("Agent cache cleared")


# Global factory instance
_agent_factory: SimplifiedAgentFactory | None = None


def get_agent_factory() -> SimplifiedAgentFactory:
    """Get the global simplified agent factory instance."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = SimplifiedAgentFactory()
    return _agent_factory


def create_agent(model_id: str, **kwargs) -> Agent:
    """
    Convenience function to create a PydanticAI Agent.

    Args:
        model_id: The model ID
        **kwargs: Additional Agent parameters

    Returns:
        Configured PydanticAI Agent
    """
    return get_agent_factory().create_agent(model_id, **kwargs)


def get_available_models() -> list[str]:
    """
    Convenience function to get available models.

    Returns:
        List of available model IDs
    """
    return get_agent_factory().get_available_models()


def list_all_models() -> list[str]:
    """
    廃止済みモデルを含む全モデル ID のリストを返す。

    Returns:
        全モデル ID のリスト
    """
    return get_agent_factory().list_all_models()


def is_model_deprecated(model_id: str) -> bool:
    """
    指定モデルが廃止済みかどうかを確認する。

    Args:
        model_id: 確認するモデルの ID

    Returns:
        廃止済みの場合 True
    """
    return get_agent_factory().is_model_deprecated(model_id)
