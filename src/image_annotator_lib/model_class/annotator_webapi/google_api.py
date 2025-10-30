from typing import Any, Self, override

from PIL import Image
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from ...core.base import WebApiBaseAnnotator
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin
from ...core.types import TaskCapability, UnifiedAnnotationResult
from ...core.utils import logger


class GoogleApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Provider-level PydanticAI Google Gemini アノテーター実装

    Provider-level アーキテクチャによる効率的なリソース共有で
    Google Gemini API と構造化出力を統合する。

    Note:
        Phase 1B: Config Object統合
    """

    def __init__(self, model_name: str, config=None):
        """Google アノテーターをmodel_nameで初期化する

        Args:
            model_name: 設定ファイルからのモデル名
            config: WebAPIModelConfig (Phase 1B DI)。Noneの場合、後方互換フォールバック。
        """
        WebApiBaseAnnotator.__init__(self, model_name, config=config)
        PydanticAIAnnotatorMixin.__init__(self, model_name, config=config)
        # 設定を初期化時に読み込む
        self._load_configuration()

    @override
    def __enter__(self) -> Self:
        """コンテキストマネージャーエントリ - Provider-level Agent準備"""
        logger.info(f"Provider-level Google アノテーター '{self.model_name}' のコンテキストに入ります...")

        try:
            self._setup_agent()
            logger.info(f"Provider-level Google Agent 準備完了 (model: {self.api_model_id})")
        except Exception as e:
            logger.error(f"Provider-level Google Agent 準備エラー: {e}")
            raise

        return self

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストマネージャー終了 - Provider-levelで管理されるため何もしない"""
        # Provider-levelで管理されるため、個別のリソース解放は不要
        logger.debug("Provider-level Google Agent コンテキスト終了")

    def run_with_model(self, images: list[Image.Image], model_id: str) -> list[UnifiedAnnotationResult]:
        """指定されたモデルIDで推論を実行する(Provider-level実装、UnifiedAnnotationResult対応)"""
        logger.debug(f"Google API 推論実行: model={model_id}, images={len(images)}")

        binary_contents = self._preprocess_images_to_binary(images)
        from ...core.utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        results: list[UnifiedAnnotationResult] = []

        for binary_content in binary_contents:
            try:
                self._wait_for_rate_limit()

                # PydanticAI Agent で推論実行(model override付き)
                response_content = self._run_inference_with_model(binary_content, model_id)

                # AnnotationSchemaからUnifiedAnnotationResultに変換
                if response_content:
                    # raw_outputの安全な処理（テスト時のMagicMock対策）
                    raw_output = None
                    try:
                        # MagicMock検出（unittest.mockのMagicMockオブジェクト）
                        if str(type(response_content)).find("MagicMock") != -1:
                            raw_output = {"mock_type": "MagicMock", "mock_str": str(response_content)}
                        elif hasattr(response_content, "model_dump") and callable(
                            response_content.model_dump
                        ):
                            raw_output = response_content.model_dump()
                        elif hasattr(response_content, "__dict__"):
                            # 通常のオブジェクトの場合（辞書変換試行）
                            try:
                                raw_output = dict(response_content.__dict__)
                            except Exception:
                                raw_output = {
                                    "object_type": str(type(response_content)),
                                    "content": str(response_content),
                                }
                        else:
                            raw_output = {"fallback_content": str(response_content)}
                    except Exception:
                        # どんなエラーでも安全にフォールバック
                        raw_output = {
                            "error": "Failed to serialize response",
                            "type": str(type(response_content)),
                        }

                    result = UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        tags=response_content.tags if TaskCapability.TAGS in capabilities else None,
                        captions=response_content.captions
                        if TaskCapability.CAPTIONS in capabilities
                        else None,
                        scores={"score": response_content.score}
                        if TaskCapability.SCORES in capabilities and response_content.score
                        else None,
                        provider_name="google",
                        framework="api",
                        raw_output=raw_output,
                    )
                    results.append(result)
                else:
                    results.append(
                        UnifiedAnnotationResult(
                            model_name=self.model_name,
                            capabilities=capabilities,
                            error="Empty response from API",
                            provider_name="google",
                            framework="api",
                        )
                    )

            except ModelHTTPError as e:
                # PydanticAI統一HTTPエラー処理
                error_message = f"Google HTTP {e.status_code}: {e.body or str(e)}"
                logger.error(f"Google API 推論エラー: {error_message}")
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=error_message,
                        provider_name="google",
                        framework="api",
                    )
                )

            except UnexpectedModelBehavior as e:
                # PydanticAI統一モデル動作エラー処理
                error_message = f"Google API Error: Unexpected model behavior: {e!s}"
                logger.error(f"Google API 推論エラー: {error_message}")
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=error_message,
                        provider_name="google",
                        framework="api",
                    )
                )

            except Exception as e:
                # その他の予期しないエラー
                error_message = f"Google API Error: {e!s}"
                logger.error(f"Google API 推論エラー: {error_message}")
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=error_message,
                        provider_name="google",
                        framework="api",
                    )
                )

        return results

    @override
    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Provider-levelでは前処理は不要"""
        return images

    @override
    def _run_inference(self, processed: list[Image.Image]) -> list[UnifiedAnnotationResult]:
        """Provider Managerを通して推論実行（UnifiedAnnotationResult対応）"""
        if not self.api_model_id:
            from ...core.utils import get_model_capabilities

            capabilities = get_model_capabilities(self.model_name)
            return [
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    error=f"Model {self.model_name} has no api_model_id configured",
                    provider_name="google",
                    framework="api",
                )
            ] * len(processed)

        # Provider-level実行に委譲
        return self.run_with_model(processed, self.api_model_id)

    @override
    def _format_predictions(
        self, raw_outputs: list[UnifiedAnnotationResult]
    ) -> list[UnifiedAnnotationResult]:
        """Provider-levelでは整形済みのため変更不要（UnifiedAnnotationResult対応）"""
        return raw_outputs

    @override
    def _generate_tags(self, formatted_output: UnifiedAnnotationResult) -> list[str]:
        """整形済み出力からタグリストを生成（UnifiedAnnotationResult対応）"""
        if formatted_output.error:
            return []

        if formatted_output.tags:
            return formatted_output.tags

        return []
