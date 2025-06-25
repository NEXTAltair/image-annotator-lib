"""PydanticAI Provider-level factory for efficient instance management."""

import asyncio
from io import BytesIO
from typing import Dict, Optional, Any, ClassVar
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models import infer_model
from pydantic_ai.providers import infer_provider
from PIL import Image

from .config import config_registry
from .types import AnnotationSchema, RawOutput
from .utils import logger
from .webapi_agent_cache import WebApiAgentCache, create_cache_key, create_config_hash
from ..model_class.annotator_webapi.webapi_shared import BASE_PROMPT


class PydanticAIProviderFactory:
    """Provider-level PydanticAI Agent factory for efficient resource sharing"""
    
    _providers: ClassVar[Dict[str, Any]] = {}
    _agent_cache: ClassVar[WebApiAgentCache] = WebApiAgentCache()
    
    @classmethod
    def get_provider(cls, provider_name: str, **provider_kwargs) -> Any:
        """Get or create provider instance for the given provider name"""
        provider_key = f"{provider_name}:{id(provider_kwargs)}"
        
        if provider_key not in cls._providers:
            provider_cls = infer_provider(provider_name)
            cls._providers[provider_key] = provider_cls(**provider_kwargs)
            logger.debug(f"Created new {provider_name} provider instance: {provider_key}")
        
        return cls._providers[provider_key]
    
    @classmethod 
    def create_agent(
        cls, 
        model_name: str, 
        api_model_id: str,
        api_key: str,
        config_hash: Optional[str] = None
    ) -> Agent:
        """Create PydanticAI Agent with provider reuse and caching"""
        
        # Use PydanticAI's built-in model inference
        if ":" not in api_model_id:
            # Auto-detect provider from model name
            if api_model_id.startswith(('gpt', 'o1', 'o3')):
                full_model_name = f"openai:{api_model_id}"
            elif api_model_id.startswith('claude'):
                full_model_name = f"anthropic:{api_model_id}"
            elif api_model_id.startswith('gemini'):
                full_model_name = f"google:{api_model_id}"
            else:
                full_model_name = api_model_id
        else:
            full_model_name = api_model_id
            
        # Use PydanticAI's native model inference
        model = infer_model(full_model_name)
        
        # Extract provider name from model
        provider_name = model.system
        
        # Create or reuse provider
        provider_kwargs = {"api_key": api_key}
        provider = cls.get_provider(provider_name, **provider_kwargs)
        
        # Override model's provider with our shared instance
        model._provider = provider
        
        # Create Agent with shared provider
        agent = Agent(
            model=model,
            output_type=AnnotationSchema,
            system_prompt=BASE_PROMPT
        )
        
        return agent
    
    @classmethod
    def get_cached_agent(
        cls,
        model_name: str,
        api_model_id: str, 
        api_key: str,
        config_data: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """Get cached Agent or create new one with provider sharing"""
        
        # Create cache key and config hash
        provider_name = cls._extract_provider_name(api_model_id)
        cache_key = create_cache_key(model_name, provider_name, api_model_id)
        config_hash = create_config_hash(config_data or {})
        
        def creator_func() -> Agent:
            # Handle OpenRouter special case
            if api_model_id.startswith("openrouter:"):
                return cls.create_openrouter_agent(model_name, api_model_id, api_key, config_data)
            else:
                return cls.create_agent(model_name, api_model_id, api_key, config_hash)
        
        return cls._agent_cache.get_agent(cache_key, creator_func, config_hash)
    
    @classmethod
    def create_openrouter_agent(
        cls,
        model_name: str,
        api_model_id: str,
        api_key: str,
        config_data: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """Create OpenRouter-specific Agent with custom headers"""
        
        # Extract actual model ID from openrouter: prefix
        actual_model_id = api_model_id[11:]  # Remove "openrouter:" prefix
        full_model_name = f"openai:{actual_model_id}"  # Use OpenAI provider
        
        # Setup OpenRouter specific provider kwargs
        provider_kwargs = {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1"
        }
        
        # Add OpenRouter custom headers if present in config
        if config_data:
            default_headers = {}
            if config_data.get("referer"):
                default_headers["HTTP-Referer"] = config_data["referer"]
            if config_data.get("app_name"):
                default_headers["X-Title"] = config_data["app_name"]
            
            if default_headers:
                provider_kwargs["default_headers"] = default_headers
        
        # Use PydanticAI's native model inference
        model = infer_model(full_model_name)
        
        # Create or reuse OpenAI provider with OpenRouter settings
        provider = cls.get_provider("openai", **provider_kwargs)
        
        # Override model's provider with our shared instance
        model._provider = provider
        
        # Create Agent with shared provider
        agent = Agent(
            model=model,
            output_type=AnnotationSchema,
            system_prompt=BASE_PROMPT
        )
        
        return agent
    
    @classmethod
    def _extract_provider_name(cls, api_model_id: str) -> str:
        """Extract provider name from model ID"""
        if ":" in api_model_id:
            return api_model_id.split(":", 1)[0]
        
        # Auto-detect provider
        if api_model_id.startswith(('gpt', 'o1', 'o3')):
            return 'openai'
        elif api_model_id.startswith('claude'):
            return 'anthropic'
        elif api_model_id.startswith('gemini'):
            return 'google'
        else:
            return 'unknown'


class PydanticAIAnnotatorMixin:
    """Mixin for PydanticAI-based annotators with provider sharing"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.agent: Optional[Agent] = None
        self.api_key: Optional[SecretStr] = None
        self.api_model_id: Optional[str] = None
    
    def _load_configuration(self):
        """Load configuration from registry"""
        self.api_key = SecretStr(config_registry.get(self.model_name, "api_key", default=""))
        self.api_model_id = config_registry.get(self.model_name, "api_model_id", default=None)
        
        if not self.api_key.get_secret_value():
            provider_name = self._get_provider_name()
            raise ValueError(f"{provider_name} API キーが設定されていません")
        if not self.api_model_id:
            provider_name = self._get_provider_name()
            raise ValueError(f"{provider_name} API モデルIDが設定されていません")
    
    def _get_provider_name(self) -> str:
        """Get provider name for this annotator"""
        if self.api_model_id:
            return PydanticAIProviderFactory._extract_provider_name(self.api_model_id)
        return "Unknown"
    
    def _get_config_hash(self) -> str:
        """Generate configuration hash"""
        config_data = {
            "model_id": self.api_model_id,
            "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
            "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
        }
        return create_config_hash(config_data)
    
    def _setup_agent(self):
        """Setup PydanticAI Agent with provider sharing"""
        self._load_configuration()
        
        config_data = {
            "model_id": self.api_model_id,
            "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
            "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
        }
        
        self.agent = PydanticAIProviderFactory.get_cached_agent(
            model_name=self.model_name,
            api_model_id=self.api_model_id,
            api_key=self.api_key.get_secret_value(),
            config_data=config_data
        )
    
    def _preprocess_images_to_binary(self, images: list[Image.Image]) -> list[BinaryContent]:
        """Convert PIL Images to PydanticAI BinaryContent format"""
        binary_contents = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="WEBP")
            binary_content = BinaryContent(data=buffered.getvalue(), media_type="image/webp")
            binary_contents.append(binary_content)
        return binary_contents
    
    def _run_inference_with_model(
        self, 
        binary_content: BinaryContent,
        override_model_id: Optional[str] = None
    ) -> AnnotationSchema:
        """Run inference with optional model override"""
        return asyncio.run(self._run_inference_async(binary_content, override_model_id))
    
    async def _run_inference_async(
        self, 
        binary_content: BinaryContent,
        override_model_id: Optional[str] = None
    ) -> AnnotationSchema:
        """Async inference with optional model override"""
        
        # If model override is requested, create temporary agent
        if override_model_id and override_model_id != self.api_model_id:
            temp_agent = PydanticAIProviderFactory.create_agent(
                model_name=f"{self.model_name}_temp",
                api_model_id=override_model_id,
                api_key=self.api_key.get_secret_value()
            )
            agent_to_use = temp_agent
            logger.debug(f"Using temporary agent with model: {override_model_id}")
        else:
            agent_to_use = self.agent
        
        # Get model parameters
        temperature = config_registry.get(self.model_name, "temperature", default=0.7)
        max_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)
        
        model_params = {
            "temperature": float(temperature) if temperature is not None else 0.7,
            "max_tokens": int(max_tokens) if max_tokens is not None else 1800,
        }
        
        # Run inference
        result = await agent_to_use.run(
            user_prompt="この画像を詳細に分析して、タグとキャプションを生成してください。",
            message_history=[binary_content],
            model_settings=model_params,
        )
        
        return result.data