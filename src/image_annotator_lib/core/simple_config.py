"""Simplified configuration management for PydanticAI models."""

from pathlib import Path
from typing import Any, Dict

import toml

from .constants import CONFIG_DIR
from .utils import logger

# Simplified config file path
MODEL_SETTINGS_PATH = CONFIG_DIR / "model_settings.toml"


class SimpleModelConfig:
    """Simplified configuration manager for PydanticAI models."""
    
    def __init__(self):
        self._config_cache: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load the simplified model settings."""
        try:
            if MODEL_SETTINGS_PATH.exists():
                with open(MODEL_SETTINGS_PATH, "r", encoding="utf-8") as f:
                    self._config_cache = toml.load(f)
                logger.info(f"Loaded model settings from {MODEL_SETTINGS_PATH}")
            else:
                logger.warning(f"Model settings file not found: {MODEL_SETTINGS_PATH}")
                self._config_cache = {"global_defaults": {}, "model_overrides": {}}
        except Exception as e:
            logger.error(f"Failed to load model settings: {e}")
            self._config_cache = {"global_defaults": {}, "model_overrides": {}}
    
    def get_model_settings(self, model_id: str) -> Dict[str, Any]:
        """
        Get combined settings for a specific model.
        
        Args:
            model_id: The model ID (e.g., "google/gemini-2.5-pro-preview-03-25")
            
        Returns:
            Dictionary with combined default and model-specific settings
        """
        # Start with global defaults
        settings = self._config_cache.get("global_defaults", {}).copy()
        
        # Apply model-specific overrides
        overrides = self._config_cache.get("model_overrides", {}).get(model_id, {})
        settings.update(overrides)
        
        logger.debug(f"Settings for {model_id}: {settings}")
        return settings
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get the global default settings."""
        return self._config_cache.get("global_defaults", {}).copy()
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()


# Global instance
_simple_config: SimpleModelConfig | None = None


def get_simple_config() -> SimpleModelConfig:
    """Get the global simple configuration instance."""
    global _simple_config
    if _simple_config is None:
        _simple_config = SimpleModelConfig()
    return _simple_config


def get_model_settings(model_id: str) -> Dict[str, Any]:
    """
    Convenience function to get model settings.
    
    Args:
        model_id: The model ID
        
    Returns:
        Dictionary with model settings
    """
    return get_simple_config().get_model_settings(model_id)


def get_default_settings() -> Dict[str, Any]:
    """
    Convenience function to get default settings.
    
    Returns:
        Dictionary with default settings
    """
    return get_simple_config().get_default_settings()