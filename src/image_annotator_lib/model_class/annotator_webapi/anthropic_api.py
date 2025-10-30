import base64
from io import BytesIO
from typing import override

from PIL import Image
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from ...core.base import WebApiBaseAnnotator
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin, _is_test_environment
from ...core.types import TaskCapability, UnifiedAnnotationResult
from ...core.utils import logger
from ...exceptions.errors import WebApiError


class AnthropicApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Anthropic Claude API を使用するアノテーター (Provider-level PydanticAI版)

    Note:
        Phase 1B: Config Object統合
        - WebAPIModelConfig経由でAPI関連設定を注入
        - 後方互換のためconfig=Noneをサポート
    """

    def __init__(self, model_name: str, config=None):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
            config: WebAPIModelConfig (Phase 1B DI)。Noneの場合、後方互換フォールバック。
        """
        WebApiBaseAnnotator.__init__(self, model_name, config=config)
        PydanticAIAnnotatorMixin.__init__(self, model_name, config=config)
        # 設定を初期化時に読み込む
        self._load_configuration()
        # テスト環境ではAPIキー検証をスキップ
        if not _is_test_environment() and (not self.api_key or not self.api_key.get_secret_value()):
            raise WebApiError("APIキーが設定されていません。", provider_name="Anthropic")

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self._setup_agent()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        # Agent は Provider Factory で管理されるため、ここでは何もしない
        pass

    def run_with_model(self, images: list[Image.Image], model_id: str) -> list[UnifiedAnnotationResult]:
        """指定されたモデルIDで推論を実行する（UnifiedAnnotationResult対応）"""
        if not self.agent:
            raise WebApiError(
                "Agent が初期化されていません。コンテキストマネージャーを使用してください。",
                provider_name="Anthropic",
            )

        binary_contents = self._preprocess_images_to_binary(images)
        from ...core.utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        results: list[UnifiedAnnotationResult] = []

        for binary_content in binary_contents:
            try:
                self._wait_for_rate_limit()

                # The result from agent.run is a ModelResponse object.
                # The actual content is in the first part of the response.
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
                        provider_name="anthropic",
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
                            provider_name="anthropic",
                            framework="api",
                        )
                    )

            except ModelHTTPError as e:
                # PydanticAI統一HTTPエラー処理
                error_message = f"Anthropic HTTP {e.status_code}: {e.body or str(e)}"
                logger.error(f"Anthropic API 推論エラー: {error_message}")
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=error_message,
                        provider_name="anthropic",
                        framework="api",
                    )
                )

            except UnexpectedModelBehavior as e:
                # PydanticAI統一モデル動作エラー処理
                error_message = f"Anthropic API Error: Unexpected model behavior: {e!s}"
                logger.error(f"Anthropic API 推論エラー: {error_message}")
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=error_message,
                        provider_name="anthropic",
                        framework="api",
                    )
                )

            except Exception as e:
                # その他の予期しないエラー
                error_message = f"Anthropic API Error: {e!s}"
                logger.error(f"Anthropic API 推論エラー: {error_message}")
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=error_message,
                        provider_name="anthropic",
                        framework="api",
                    )
                )

        return results

    @override
    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[UnifiedAnnotationResult]:
        """PydanticAI Agent を使用して推論を実行する (デフォルトモデル、UnifiedAnnotationResult対応)"""
        # PIL.Image リストに変換
        try:
            pil_images = []
            for item in processed_images:
                if isinstance(item, str):
                    # Base64 文字列の場合
                    image_data = base64.b64decode(item)
                    pil_image = Image.open(BytesIO(image_data))
                    pil_images.append(pil_image)
                elif isinstance(item, bytes):
                    # バイナリデータの場合
                    pil_image = Image.open(BytesIO(item))
                    pil_images.append(pil_image)
                else:
                    # すでに PIL.Image の場合
                    pil_images.append(item)
        except Exception as e:
            logger.error(f"画像前処理エラー: {e}")
            from ...core.utils import get_model_capabilities

            capabilities = get_model_capabilities(self.model_name)
            return [
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    error=f"画像前処理エラー: {e}",
                    provider_name="anthropic",
                    framework="api",
                )
            ] * len(processed_images)

        # デフォルトモデルで実行
        return self.run_with_model(pil_images, self.api_model_id)
