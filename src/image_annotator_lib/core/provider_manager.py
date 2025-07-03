"""Provider-level instance manager for efficient PydanticAI usage."""

import asyncio
from typing import Any, ClassVar

from PIL import Image

from ..exceptions.errors import WebApiError
from .config import config_registry
from .types import AnnotationResult, RawOutput
from .utils import calculate_phash, logger


class ProviderManager:
    """Manages provider-level instances for efficient PydanticAI usage"""

    _provider_instances: ClassVar[dict[str, Any]] = {}

    @classmethod
    def get_provider_instance(cls, provider_name: str) -> Any:
        """Get or create provider-level instance"""

        if provider_name not in cls._provider_instances:
            # Create new provider-level instance
            if provider_name == "anthropic":
                cls._provider_instances[provider_name] = AnthropicProviderInstance()
            elif provider_name == "openai":
                cls._provider_instances[provider_name] = OpenAIProviderInstance()
            elif provider_name == "openrouter":
                cls._provider_instances[provider_name] = OpenRouterProviderInstance()
            elif provider_name == "google":
                cls._provider_instances[provider_name] = GoogleProviderInstance()
            else:
                raise WebApiError(f"Unsupported provider: {provider_name}")

            logger.debug(f"Created new provider instance: {provider_name}")

        return cls._provider_instances[provider_name]

    @classmethod
    def run_inference_with_model(
        cls, model_name: str, images_list: list[Image.Image], api_model_id: str
    ) -> dict[str, AnnotationResult]:
        """Run inference with specified model ID using provider sharing

        Returns:
            Dict mapping image phash to AnnotationResult
        """

        # Determine provider from model configuration or model ID
        provider_name = cls._determine_provider(model_name, api_model_id)

        # Get provider instance
        provider_instance = cls.get_provider_instance(provider_name)

        # Execute inference
        raw_outputs = provider_instance.run_with_model(model_name, images_list, api_model_id)

        # Convert to phash-based dict format expected by tests
        results = {}
        for i, raw_output in enumerate(raw_outputs):
            # Generate phash for each image
            image_phash = calculate_phash(images_list[i]) if i < len(images_list) else f"unknown_image_{i}"

            # Convert RawOutput to AnnotationResult format
            if raw_output.get("error"):
                annotation_result = AnnotationResult(
                    phash=image_phash, tags=[], formatted_output=None, error=raw_output["error"]
                )
            else:
                response = raw_output.get("response")
                if response:
                    tags = getattr(response, "tags", []) if hasattr(response, "tags") else []
                    annotation_result = AnnotationResult(
                        phash=image_phash, tags=tags, formatted_output=response, error=None
                    )
                else:
                    annotation_result = AnnotationResult(
                        phash=image_phash, tags=[], formatted_output=None, error="No response from provider"
                    )

            results[image_phash] = annotation_result

        return results

    @classmethod
    def _run_agent_safely(cls, agent, binary_content, api_model_id: str):
        """Safely run PydanticAI agent with simplified event loop management"""
        logger.debug(f"Attempting to run agent with model_id: {api_model_id}")
        logger.debug(f"Agent type: {type(agent)}")
        logger.debug(f"Binary content type: {type(binary_content)}")
        logger.debug(f"Binary content media_type: {getattr(binary_content, 'media_type', 'unknown')}")
        
        try:
            # Simple approach: always use run_sync and handle any event loop issues
            # For PydanticAI, images are passed as a list directly to run_sync
            result = agent.run_sync(
                [binary_content],  # Pass as direct list, not as message_history
                model=api_model_id
            )
            logger.debug(f"Agent execution successful. Result type: {type(result)}")
            return result
        except RuntimeError as e:
            if "Event loop is closed" in str(e) or "asyncio" in str(e):
                # Try to create a new event loop and run async
                try:
                    import concurrent.futures
                    
                    def run_with_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                agent.run(
                                    [binary_content],  # Pass as direct list
                                    model=api_model_id
                                )
                            )
                        finally:
                            new_loop.close()
                            asyncio.set_event_loop(None)
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_with_new_loop)
                        return future.result(timeout=60)
                except Exception as fallback_error:
                    logger.error(f"Fallback agent execution failed: {fallback_error}", exc_info=True)
                    raise RuntimeError(f"Both sync and async execution failed: {e}, {fallback_error}")
            else:
                raise
        except Exception as e:
            # Enhanced error reporting for debugging
            error_context = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "api_model_id": api_model_id,
                "agent_type": type(agent).__name__,
                "agent_model": getattr(agent, 'model', 'unknown'),
                "binary_content_size": len(getattr(binary_content, 'data', b'')),
                "media_type": getattr(binary_content, 'media_type', 'unknown')
            }
            logger.error(f"Agent execution error: error_type={type(e).__name__}, message={str(e)}, api_model_id={api_model_id}, agent_type={type(agent).__name__}", exc_info=True)
            raise

    @classmethod
    def _determine_provider(cls, model_name: str, api_model_id: str) -> str:
        """Determine provider from model configuration or model ID"""

        # First check explicit provider configuration
        provider = config_registry.get(model_name, "provider")
        if provider:
            return provider.lower()

        # Auto-detect from model ID
        if ":" in api_model_id:
            return api_model_id.split(":", 1)[0]

        # Auto-detect from model name patterns
        if api_model_id.startswith(("gpt", "o1", "o3")):
            return "openai"
        elif api_model_id.startswith("claude"):
            return "anthropic"
        elif api_model_id.startswith("gemini"):
            return "google"
        else:
            # Default fallback based on model configuration structure
            if config_registry.get(model_name, "anthropic_api_key"):
                return "anthropic"
            elif config_registry.get(model_name, "openai_api_key"):
                return "openai"
            elif config_registry.get(model_name, "google_api_key"):
                return "google"
            else:
                return "openai"  # Default fallback


class ProviderInstanceBase:
    """Base class for provider instances"""

    def __init__(self):
        self._active_contexts = {}

    def run_with_model(
        self, model_name: str, images_list: list[Image.Image], api_model_id: str
    ) -> list[RawOutput]:
        """Run inference with specified model ID"""

        # Get or create context-managed annotator for this model_name
        if model_name not in self._active_contexts:
            annotator = self._create_annotator(model_name)
            context = annotator.__enter__()
            self._active_contexts[model_name] = (annotator, context)

        _, context = self._active_contexts[model_name]

        # Execute inference with specified model
        return context.run_with_model(images_list, api_model_id)

    def _create_annotator(self, model_name: str):
        """Create annotator instance - to be implemented by subclasses"""
        raise NotImplementedError

    def cleanup_context(self, model_name: str):
        """Clean up context for specific model"""
        if model_name in self._active_contexts:
            annotator, context = self._active_contexts[model_name]
            try:
                annotator.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during context cleanup for {model_name}: {e}")
            del self._active_contexts[model_name]


class AnthropicProviderInstance(ProviderInstanceBase):
    """Anthropic provider instance using PydanticAI Factory directly"""

    def run_with_model(self, model_name: str, images_list: list[Image.Image], api_model_id: str) -> list[dict]:
        """Direct execution using PydanticAI Factory"""
        from .pydantic_ai_factory import PydanticAIProviderFactory
        from .config import config_registry
        
        try:
            # API key を取得
            api_key = config_registry.get(model_name, "api_key", default="")
            
            # テスト環境では APIキーチェックをスキップ
            from .pydantic_ai_factory import _is_test_environment
            is_test_env = _is_test_environment()
            logger.debug(f"Test environment detected: {is_test_env}")
            if not api_key and not is_test_env:
                return [{"error": "Anthropic API key not configured"}] * len(images_list)
            
            # Agent を取得
            agent = PydanticAIProviderFactory.get_cached_agent(
                model_name=model_name,
                api_model_id=api_model_id,
                api_key=api_key
            )
            
            # 推論実行
            results = []
            for image in images_list:
                try:
                    # 画像を適切な形式に変換
                    from io import BytesIO
                    from pydantic_ai import BinaryContent
                    
                    buffered = BytesIO()
                    image.save(buffered, format="WEBP")
                    binary_content = BinaryContent(data=buffered.getvalue(), media_type="image/webp")
                    
                    # Safe sync execution with event loop management
                    result = ProviderManager._run_agent_safely(
                        agent, binary_content, api_model_id
                    )
                    results.append({"response": result.data})
                except Exception as e:
                    # Enhanced error logging for debugging
                    error_details = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "model_name": model_name,
                        "api_model_id": api_model_id,
                        "provider": "anthropic"
                    }
                    logger.error(f"Anthropic provider error details: error_type={type(e).__name__}, message={str(e)}, model={model_name}, api_id={api_model_id}", exc_info=True)
                    results.append({"error": f"Anthropic API Error: {e}"})
            
            return results
            
        except Exception as e:
            logger.error(f"Anthropic provider error: {e}", exc_info=True)
            return [{"error": f"Anthropic provider failed: {e}"}] * len(images_list)

    def _create_annotator(self, model_name: str):
        # Backward compatibility - not used in direct execution
        from ..model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
        return AnthropicApiAnnotator(model_name)


class OpenAIProviderInstance(ProviderInstanceBase):
    """OpenAI provider instance using PydanticAI Factory directly"""

    def run_with_model(self, model_name: str, images_list: list[Image.Image], api_model_id: str) -> list[dict]:
        """Direct execution using PydanticAI Factory"""
        from .pydantic_ai_factory import PydanticAIProviderFactory
        from .config import config_registry
        
        try:
            # API key を取得
            api_key = config_registry.get(model_name, "api_key", default="")
            
            # テスト環境では APIキーチェックをスキップ
            from .pydantic_ai_factory import _is_test_environment
            is_test_env = _is_test_environment()
            logger.debug(f"Test environment detected: {is_test_env}")
            if not api_key and not is_test_env:
                return [{"error": "OpenAI API key not configured"}] * len(images_list)
            
            # Agent を取得
            agent = PydanticAIProviderFactory.get_cached_agent(
                model_name=model_name,
                api_model_id=api_model_id,
                api_key=api_key
            )
            
            # 推論実行
            results = []
            for image in images_list:
                try:
                    # 画像を適切な形式に変換
                    from io import BytesIO
                    from pydantic_ai import BinaryContent
                    
                    buffered = BytesIO()
                    image.save(buffered, format="WEBP")
                    binary_content = BinaryContent(data=buffered.getvalue(), media_type="image/webp")
                    
                    # Safe sync execution with event loop management
                    result = ProviderManager._run_agent_safely(
                        agent, binary_content, api_model_id
                    )
                    results.append({"response": result.data})
                except Exception as e:
                    # Enhanced error logging for debugging
                    error_details = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "model_name": model_name,
                        "api_model_id": api_model_id,
                        "provider": "openai"
                    }
                    logger.error(f"OpenAI provider error details: error_type={type(e).__name__}, message={str(e)}, model={model_name}, api_id={api_model_id}", exc_info=True)
                    results.append({"error": f"OpenAI API Error: {e}"})
            
            return results
            
        except Exception as e:
            logger.error(f"OpenAI provider error: {e}", exc_info=True)
            return [{"error": f"OpenAI provider failed: {e}"}] * len(images_list)

    def _create_annotator(self, model_name: str):
        # Backward compatibility - not used in direct execution
        from ..model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
        return OpenAIApiAnnotator(model_name)


class OpenRouterProviderInstance(ProviderInstanceBase):
    """OpenRouter provider instance using PydanticAI Factory directly"""

    def run_with_model(self, model_name: str, images_list: list[Image.Image], api_model_id: str) -> list[dict]:
        """Direct execution using PydanticAI Factory"""
        from .pydantic_ai_factory import PydanticAIProviderFactory
        from .config import config_registry
        
        try:
            # API key を取得
            api_key = config_registry.get(model_name, "api_key", default="")
            
            # テスト環境では APIキーチェックをスキップ
            from .pydantic_ai_factory import _is_test_environment
            is_test_env = _is_test_environment()
            logger.debug(f"Test environment detected: {is_test_env}")
            if not api_key and not is_test_env:
                return [{"error": "OpenRouter API key not configured"}] * len(images_list)
            
            # Agent を取得
            agent = PydanticAIProviderFactory.get_cached_agent(
                model_name=model_name,
                api_model_id=api_model_id,
                api_key=api_key
            )
            
            # 推論実行
            results = []
            for image in images_list:
                try:
                    # 画像を適切な形式に変換
                    from io import BytesIO
                    from pydantic_ai import BinaryContent
                    
                    buffered = BytesIO()
                    image.save(buffered, format="WEBP")
                    binary_content = BinaryContent(data=buffered.getvalue(), media_type="image/webp")
                    
                    # Safe sync execution with event loop management
                    result = ProviderManager._run_agent_safely(
                        agent, binary_content, api_model_id
                    )
                    results.append({"response": result.data})
                except Exception as e:
                    # Enhanced error logging for debugging
                    logger.error(f"OpenRouter provider error details: error_type={type(e).__name__}, message={str(e)}, model={model_name}, api_id={api_model_id}", exc_info=True)
                    results.append({"error": f"OpenRouter API Error: {e}"})
            
            return results
            
        except Exception as e:
            logger.error(f"OpenRouter provider error: {e}", exc_info=True)
            return [{"error": f"OpenRouter provider failed: {e}"}] * len(images_list)

    def _create_annotator(self, model_name: str):
        # Backward compatibility - not used in direct execution
        from ..model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        return OpenRouterApiAnnotator(model_name)


class GoogleProviderInstance(ProviderInstanceBase):
    """Google provider instance using PydanticAI Factory directly"""

    def run_with_model(self, model_name: str, images_list: list[Image.Image], api_model_id: str) -> list[dict]:
        """Direct execution using PydanticAI Factory"""
        from .pydantic_ai_factory import PydanticAIProviderFactory
        from .config import config_registry
        
        try:
            # API key を取得
            api_key = config_registry.get(model_name, "api_key", default="")
            
            # テスト環境では APIキーチェックをスキップ
            from .pydantic_ai_factory import _is_test_environment
            is_test_env = _is_test_environment()
            logger.debug(f"Test environment detected: {is_test_env}")
            if not api_key and not is_test_env:
                return [{"error": "Google API key not configured"}] * len(images_list)
            
            # Agent を取得
            agent = PydanticAIProviderFactory.get_cached_agent(
                model_name=model_name,
                api_model_id=api_model_id,
                api_key=api_key
            )
            
            # 推論実行
            results = []
            for image in images_list:
                try:
                    # 画像を適切な形式に変換
                    from io import BytesIO
                    from pydantic_ai import BinaryContent
                    
                    buffered = BytesIO()
                    image.save(buffered, format="WEBP")
                    binary_content = BinaryContent(data=buffered.getvalue(), media_type="image/webp")
                    
                    # Safe sync execution with event loop management
                    result = ProviderManager._run_agent_safely(
                        agent, binary_content, api_model_id
                    )
                    results.append({"response": result.data})
                except Exception as e:
                    # Enhanced error logging for debugging
                    logger.error(f"Google provider error details: error_type={type(e).__name__}, message={str(e)}, model={model_name}, api_id={api_model_id}", exc_info=True)
                    results.append({"error": f"Google API Error: {e}"})
            
            return results
            
        except Exception as e:
            logger.error(f"Google provider error: {e}", exc_info=True)
            return [{"error": f"Google provider failed: {e}"}] * len(images_list)

    def _create_annotator(self, model_name: str):
        # Backward compatibility - not used in direct execution
        from ..model_class.annotator_webapi.google_api import GoogleApiAnnotator
        return GoogleApiAnnotator(model_name)
