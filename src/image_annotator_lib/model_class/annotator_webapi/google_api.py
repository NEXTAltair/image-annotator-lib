from typing import Any, Self, override

from PIL import Image

from ...core.base import WebApiBaseAnnotator
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin
from ...core.types import RawOutput
from ...core.utils import logger
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from ...exceptions.errors import WebApiError


class GoogleApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Provider-level PydanticAI Google Gemini アノテーター実装

    Provider-level アーキテクチャによる効率的なリソース共有で
    Google Gemini API と構造化出力を統合する。
    """

    def __init__(self, model_name: str):
        """Google アノテーターをmodel_nameで初期化する

        Args:
            model_name: 設定ファイルからのモデル名
        """
        WebApiBaseAnnotator.__init__(self, model_name)
        PydanticAIAnnotatorMixin.__init__(self, model_name)
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

    def run_with_model(self, images: list[Image.Image], model_id: str) -> list[RawOutput]:
        """指定されたモデルIDで推論を実行する(Provider-level実装)"""
        logger.debug(f"Google API 推論実行: model={model_id}, images={len(images)}")

        try:
            # 画像をBinaryContentに変換
            binary_contents = self._preprocess_images_to_binary(images)

            results = []
            for binary_content in binary_contents:
                try:
                    self._wait_for_rate_limit()

                    # PydanticAI Agent で推論実行(model override付き)
                    annotation = self._run_inference_with_model(binary_content, model_id)
                    results.append({"response": annotation, "error": None})

                except ModelHTTPError as e:
                    # PydanticAI統一HTTPエラー処理
                    error_message = f"Google HTTP {e.status_code}: {e.response_body or str(e)}"
                    logger.error(f"Google API 推論エラー: {error_message}")
                    results.append({"response": None, "error": error_message})

                except UnexpectedModelBehavior as e:
                    # PydanticAI統一モデル動作エラー処理
                    error_message = f"Google API Error: Unexpected model behavior: {str(e)}"
                    logger.error(f"Google API 推論エラー: {error_message}")
                    results.append({"response": None, "error": error_message})

                except Exception as e:
                    # その他の予期しないエラー
                    error_message = f"Google API Error: {str(e)}"
                    logger.error(f"Google API 推論エラー: {error_message}")
                    results.append({"response": None, "error": error_message})

            return results

        except Exception as e:
            logger.error(f"Google API run_with_model エラー: {e}")
            # 全画像にエラーを返す
            error_message = f"Google API Error: {str(e)}"
            return [{"response": None, "error": error_message} for _ in images]


    @override
    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Provider-levelでは前処理は不要"""
        return images

    @override
    def _run_inference(self, processed: list[Image.Image]) -> list[RawOutput]:
        """Provider Managerを通して推論実行"""
        if not self.api_model_id:
            raise ValueError(f"Model {self.model_name} has no api_model_id configured")

        # Provider-level実行に委譲
        return self.run_with_model(processed, self.api_model_id)

    @override
    def _format_predictions(self, raw_outputs: list[RawOutput]) -> list[RawOutput]:
        """Provider-levelでは整形済みのため変更不要"""
        return raw_outputs

    @override
    def _generate_tags(self, formatted_output: RawOutput) -> list[str]:
        """整形済み出力からタグリストを生成"""
        if formatted_output.get("error"):
            return []

        annotation = formatted_output.get("response")
        if annotation:
            if hasattr(annotation, "tags"):
                return annotation.tags
            elif isinstance(annotation, dict) and "tags" in annotation:
                return annotation["tags"]

        return []
