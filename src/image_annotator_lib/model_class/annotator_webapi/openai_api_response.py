from typing import Any

from PIL import Image

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.provider_manager import ProviderManager
from ...core.types import UnifiedAnnotationResult
from ...core.utils import logger


class OpenAIApiAnnotator(WebApiBaseAnnotator):
    """OpenAI API を使用するアノテーター (Plan 1 統一実装)."""

    _PROVIDER_NAME = "openai"
    _PROVIDER_DISPLAY = "OpenAI"

    def __init__(self, model_name: str, config: Any = None):
        """Initialize with model name."""
        super().__init__(model_name, config=config)

    def __enter__(self):
        """Context manager entry."""
        logger.debug(f"Entering context for OpenAI annotator: {self.model_name}")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Context manager exit."""
        pass

    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Provider-level: No preprocessing needed."""
        return images

    def _run_inference(self, processed: list[Image.Image]) -> list[UnifiedAnnotationResult]:
        """Run inference via ProviderManager."""
        api_model_id = config_registry.get(self.model_name, "api_model_id")
        if not api_model_id:
            from ...core.utils import get_model_capabilities

            capabilities = get_model_capabilities(self.model_name)
            return [
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    error=f"Model {self.model_name} has no api_model_id configured",
                    provider_name=self._PROVIDER_NAME,
                    framework="api",
                )
            ] * len(processed)

        results_dict = ProviderManager.run_inference_with_model(
            model_name=self.model_name,
            images_list=processed,
            api_model_id=api_model_id,
        )

        # Convert dict results to list format
        results = []
        for i, img in enumerate(processed):
            from ...core.utils import calculate_phash

            phash = calculate_phash(img)
            if phash in results_dict:
                results.append(results_dict[phash])
            else:
                from ...core.utils import get_model_capabilities

                capabilities = get_model_capabilities(self.model_name)
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error="No result found for image",
                        provider_name=self._PROVIDER_NAME,
                        framework="api",
                    )
                )
        return results

    def _format_predictions(
        self, raw_outputs: list[UnifiedAnnotationResult]
    ) -> list[UnifiedAnnotationResult]:
        """Predictions are already formatted."""
        return raw_outputs

    def _generate_tags(self, formatted_output: UnifiedAnnotationResult) -> list[str]:
        """Extract tags from result."""
        if formatted_output.error:
            return []
        return formatted_output.tags if formatted_output.tags else []
