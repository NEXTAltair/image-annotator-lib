"""Web API を利用するアノテーターの基底クラス。"""

import asyncio
import json
import re
import traceback
from abc import abstractmethod
from typing import Any, NoReturn, Self, override

from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import (
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    ConfigurationError,
    InsufficientCreditsError,
    WebApiError,
)
from ..config import config_registry
from ..model_factory import prepare_web_api_components
from ..types import AnnotationSchema, RawOutput, WebApiComponents, WebApiFormattedOutput
from ..utils import logger
from .annotator import BaseAnnotator


class WebApiBaseAnnotator(BaseAnnotator):
    """Web API を利用するアノテーターの基底クラス。"""

    def __init__(self, model_name: str):
        """初期化 (model_name のみ受け取るように変更)"""
        super().__init__(model_name)
        self.prompt_template = config_registry.get(
            self.model_name, "prompt_template", "Describe this image."
        )

        timeout_val: Any = config_registry.get(self.model_name, "timeout", 60)
        try:
            self.timeout = int(timeout_val) if timeout_val is not None else 60
        except (ValueError, TypeError):
            logger.warning(
                f"timeout に不正な値 {timeout_val} が設定されました。デフォルトの 60 を使用します。"
            )
            self.timeout = 60

        retry_count_val: Any = config_registry.get(self.model_name, "retry_count", 3)
        try:
            self.retry_count = int(retry_count_val) if retry_count_val is not None else 3
        except (ValueError, TypeError):
            logger.warning(
                f"retry_count に不正な値 {retry_count_val} が設定されました。デフォルトの 3 を使用します。"
            )
            self.retry_count = 3

        retry_delay_val: Any = config_registry.get(self.model_name, "retry_delay", 1.0)
        try:
            self.retry_delay = float(retry_delay_val) if retry_delay_val is not None else 1.0
        except (ValueError, TypeError):
            logger.warning(
                f"retry_delay に不正な値 {retry_delay_val} が設定されました。デフォルトの 1.0 を使用します。"
            )
            self.retry_delay = 1.0

        self.last_request_time = 0.0
        min_interval_val: Any = config_registry.get(self.model_name, "min_request_interval", 1.0)
        try:
            self.min_request_interval = float(min_interval_val) if min_interval_val is not None else 1.0
        except (ValueError, TypeError):
            logger.warning(
                f"min_request_interval に不正な値 {min_interval_val} が設定されました。デフォルトの 1.0 を使用します。"
            )
            self.min_request_interval = 1.0

        self.model_id_on_provider: str | None = None  # __enter__ で設定される
        self.api_model_id: str | None = None  # __enter__ で設定される (加工済みID)
        self.provider_name: str | None = None  # __enter__ で設定される

        self.max_output_tokens: int | None = config_registry.get(self.model_name, "max_output_tokens", 1800)
        if self.max_output_tokens is not None and not isinstance(self.max_output_tokens, int):
            logger.warning(
                f"max_output_tokens に不正な値 {self.max_output_tokens} が設定されました。None を使用します。"
            )
            self.max_output_tokens = None

        # APIキーは __enter__ で prepare_web_api_components から取得されるため、ここでは不要

        self.client: Any = None  # __enter__ で ApiClient 型に設定される
        self.components: WebApiComponents | None = None  # WebApiComponents 型を明示

    def __enter__(self) -> Self:
        """Web API コンポーネントを準備します。

        model_factory.prepare_web_api_components を呼び出して、
        APIクライアント、加工済みモデルID、プロバイダー名を取得し、
        self.components に設定します。
        """
        logger.info(f"Web API アノテーター '{self.model_name}' のコンテキストに入ります...")
        try:
            # model_factory からコンポーネントを準備
            self.components = prepare_web_api_components(self.model_name)

            # 利便性のために主要なコンポーネントをインスタンス変数にも設定
            self.client = self.components["client"]
            self.api_model_id = self.components["api_model_id"]
            # provider_name は components 経由でアクセス可能だが、変数にも設定しておく
            self.provider_name = self.components["provider_name"]

            logger.info(f"Web API コンポーネント準備完了 ({self.provider_name}, {self.api_model_id})。")

        except (ConfigurationError, ApiAuthenticationError) as e:
            logger.error(f"Web API コンポーネントの準備中に設定/認証エラーが発生: {e}")
            self.components = None
            self.client = None
            self.api_model_id = None
            self.provider_name = None  # エラー時は None
            raise  # エラーを再送出してコンテキストの失敗を通知
        except Exception as e:
            logger.exception(f"Web API コンポーネントの準備中に予期せぬエラーが発生: {e}")
            self.components = None
            self.client = None
            self.api_model_id = None
            self.provider_name = None  # エラー時は None
            raise ConfigurationError(f"Web API コンポーネント準備中の予期せぬエラー: {e}") from e

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """APIクライアントのリソースを解放 (Noneを設定) します。"""
        # self.provider_name は __enter__ で設定されるかエラーになるはず
        provider_name = getattr(self, "provider_name", self.model_name)
        if self.client:
            logger.debug(f"APIクライアントの閉鎖/リリース ({provider_name}) ...")
            # クライアントによっては close() メソッドなどが必要かもしれないが、
            # 現状のライブラリ (OpenAI, Anthropic, google.generativeai) では
            # 明示的な close は必須ではないため、参照を None にするだけで十分。
            self.client = None
            logger.debug(f"APIクライアント ({provider_name}) の参照を解放しました。")
        self.components = None  # components もクリア

    def _preprocess_images(self, images: list[Image.Image]) -> list[str] | list[bytes]:
        """画像リストを Base64 エンコードした文字列のリストに変換する"""
        import base64
        from io import BytesIO

        encoded_images = []
        for image in images:
            buffered = BytesIO()
            # 画像をWEBP形式でメモリに保存
            image.save(buffered, format="WEBP")
            # バイトデータを取得し、Base64エンコードしてUTF-8文字列にデコード
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            encoded_images.append(encoded_image)

        return encoded_images

    @abstractmethod
    def _run_inference(self, processed: list[str] | list[bytes]) -> Any:
        """Web API にリクエストを送信し、生のレスポンスを取得します。

        Args:
            processed: 前処理済みの画像データ (Base64文字列のリストまたはバイト列のリスト)。

        Returns:
            APIからの生のレスポンス。形式はAPIプロバイダによって異なります。
            通常、単一のリクエストに対するレスポンスが期待されます。
        """
        raise NotImplementedError

    def _wait_for_rate_limit(self) -> None:
        """レート制限に従ってリクエスト間隔を調整する"""
        import time

        elapsed_time = time.time() - self.last_request_time
        wait_time = self.min_request_interval - elapsed_time
        if wait_time > 0:
            logger.debug(f"レート制限のため {wait_time:.2f} 秒待機します。")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """API エラーを捕捉し、適切なカスタム例外を発生させます。

        Args:
            e: 発生した例外。

        Raises:
            ApiAuthenticationError: API認証に失敗した場合 (401)。
            InsufficientCreditsError: クレジット不足の場合 (402)。
            ApiRateLimitError: APIのレート制限に達した場合 (429)。
            ApiRequestError: リクエストの形式または内容に問題があった場合 (400)。
            ApiServerError: APIサーバーで5xx系のエラーが発生した場合。
            ApiTimeoutError: APIリクエストがタイムアウトした場合。
            WebApiError: その他のAPI関連エラーの場合。
            ConfigurationError: provider_name 属性が設定されていない場合。
        """
        error_message = str(e)
        logger.error(f"API エラーが発生しました: {error_message}")
        logger.debug(traceback.format_exc())

        # provider_name 属性の存在確認
        if not hasattr(self, "provider_name") or not self.provider_name:
            raise ConfigurationError(
                f"Annotatorクラス ({self.__class__.__name__}) に 'provider_name' 属性が設定されていません。"
            )
        provider_name = self.provider_name

        # HTTPステータスコードに基づくエラーハンドリング
        if hasattr(e, "status_code"):
            status_code = getattr(e, "status_code", 0)
            if status_code == 401:
                raise ApiAuthenticationError(provider_name=provider_name) from e
            elif status_code == 402:
                raise InsufficientCreditsError(provider_name=provider_name) from e
            elif status_code == 429:
                retry_after_str = getattr(e, "retry_after", "60")  # デフォルト60秒
                try:
                    retry_after = int(retry_after_str)
                except ValueError:
                    retry_after = 60  # パース失敗時もデフォルト値
                raise ApiRateLimitError(provider_name=provider_name, retry_after=retry_after) from e
            elif status_code == 400:
                raise ApiRequestError(error_message, provider_name=provider_name) from e
            elif 500 <= status_code < 600:
                raise ApiServerError(
                    error_message, provider_name=provider_name, status_code=status_code
                ) from e

        # タイムアウトエラーの判定を強化
        if isinstance(e, TimeoutError | asyncio.TimeoutError) or "timeout" in error_message.lower():
            raise ApiTimeoutError(provider_name=provider_name) from e

        # 上記のいずれにも当てはまらない場合、汎用のWebApiErrorを送出
        raise WebApiError(
            f"処理中に予期せぬエラーが発生しました: {error_message}", provider_name=provider_name
        ) from e

    def _parse_common_json_response(self, text_content: str | dict[str, Any]) -> WebApiFormattedOutput:
        """共通のJSONレスポンス文字列を解析し、WebApiFormattedOutputを生成するヘルパー。
        Anthropicの場合はtoolで作成されたdictなので何もせずreturnする

        Args:
            text_content: APIから返されたテキストコンテンツ。

        Returns:
            解析結果を含むWebApiFormattedOutput辞書。
            エラーが発生した場合は、errorフィールドにメッセージが含まれる。
        """
        if isinstance(text_content, dict):
            # 構造が AnnotationSchema に合うかバリデーションするのが望ましい
            # ここでは簡略化のため、dict であればそのまま通す
            # TODO: AnnotationSchema.model_validate(text_content) のようなバリデーションを追加検討
            try:
                validated_annotation = AnnotationSchema.model_validate(text_content).model_dump()
                return WebApiFormattedOutput(annotation=validated_annotation, error=None)
            except Exception as val_e:
                error_msg = f"Received dict does not match AnnotationSchema: {val_e}. Dict: {str(text_content)[:200]}"
                logger.warning(error_msg)
                return WebApiFormattedOutput(annotation=None, error=error_msg)

        logger.debug(f"_parse_common_json_response を開始: text='{text_content[:100]}...'")
        try:
            # JSON文字列を辞書にパース
            data = json.loads(text_content)

            # "Annotation" キー (Gemini) または ルートレベルの辞書 (OpenAI/Anthropic/OpenRouter) を想定
            annotation_data: dict[str, Any] | None = None
            if isinstance(data, dict):
                if "Annotation" in data and isinstance(data["Annotation"], dict):
                    annotation_data = data["Annotation"]
                    logger.debug("JSONに 'Annotation' キーが見つかりました。")
                # 'tags', 'caption', 'score' がルートレベルに存在するケースも考慮
                elif any(key in data for key in ("tags", "caption", "score")):
                    annotation_data = data
                    logger.debug("JSONのルートレベルに注釈キーが見つかりました。")
                else:
                    logger.warning("JSON内に 'Annotation' キーまたは期待されるキーが見つかりません。")
                    return WebApiFormattedOutput(
                        annotation=None,
                        error="JSON内に期待されるキー (Annotation, tags, caption, score) が見つかりません。",
                    )
            else:
                logger.warning(f"JSONデータが予期しない型 ({type(data)}) です。")
                return WebApiFormattedOutput(
                    annotation=None, error=f"JSONデータが予期しない型 ({type(data)}) です。"
                )

            if annotation_data:
                logger.debug(f"JSON解析成功。Annotation: {str(annotation_data)[:100]}...")
                return WebApiFormattedOutput(annotation=annotation_data, error=None)
            else:
                return WebApiFormattedOutput(
                    annotation=None, error="解析後、有効なAnnotationデータが見つかりませんでした。"
                )

        except json.JSONDecodeError as json_e:
            error_message = (
                f"JSON解析エラー: {json_e!s}. テキスト内容: '{text_content[:100]}...'"  # 末尾の \" を削除
            )
            logger.error(error_message)
            return WebApiFormattedOutput(annotation=None, error=error_message)
        except Exception as e:
            error_message = f"JSON解析中に予期せぬエラー: {e!s}"  # 末尾の \" を削除
            logger.exception(error_message)  # スタックトレースも記録
            return WebApiFormattedOutput(annotation=None, error=error_message)

    def _extract_tags_from_text(self, text: str) -> list[str]:
        """API レスポンス (テキスト形式) からタグリストを抽出する基本実装。

        JSON形式、またはカンマ区切りのタグリスト形式を試みます。

        Args:
            text: API から返されたテキスト応答。

        Returns:
            抽出されたタグのリスト。見つからない場合は空リスト。
        """
        logger.debug("_extract_tags_from_text を開始します。")
        tags: list[str] = []

        # 1. JSON 形式の解析を試みる
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # "tags" キーが存在し、リストまたはカンマ区切り文字列の場合
                if "tags" in data:
                    tags_data = data["tags"]
                    if isinstance(tags_data, list):
                        tags = [str(tag).strip() for tag in tags_data]  # 文字列に変換
                        logger.debug(f"JSONから {len(tags)} 個のタグを抽出しました。")
                        return tags
                    elif isinstance(tags_data, str):
                        tags = [tag.strip() for tag in tags_data.split(",") if tag.strip()]
                        logger.debug(f"JSON内のカンマ区切り文字列から {len(tags)} 個のタグを抽出しました。")
                        return tags
                # "Annotation" -> "tags" のネスト構造も考慮 (Geminiの例)
                elif (
                    "Annotation" in data
                    and isinstance(data["Annotation"], dict)
                    and "tags" in data["Annotation"]
                ):
                    tags_data = data["Annotation"]["tags"]
                    if isinstance(tags_data, list):
                        tags = [str(tag).strip() for tag in tags_data]
                        logger.debug(f"JSON (Annotation->tags) から {len(tags)} 個のタグを抽出しました。")
                        return tags
            # JSONがリスト形式で、要素が文字列の場合
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                tags = [item.strip() for item in data if item.strip()]
                logger.debug(f"JSONリストから {len(tags)} 個のタグを抽出しました。")
                return tags

        except json.JSONDecodeError:
            logger.debug("テキストは有効なJSONではありません。次の抽出方法を試みます。")
        except Exception as e:
            logger.warning(f"JSON解析中に予期せぬエラー: {e}。次の抽出方法を試みます。", exc_info=True)

        # 2. カンマ区切りテキスト形式の解析を試みる
        # "tags:" のようなプレフィックスがある場合とない場合の両方を考慮
        # より具体的にタグらしきものを抽出する正規表現
        # 例: tags: tag1, tag2, tag3 / tags: "tag1", "tag2" / tag1, tag2, ...
        patterns = [
            r"tags:?\s*\[?\"?\'?(.*?)\'?\"?\]?$",  # tags: ["tag1", "tag2"] or tags: 'tag1', 'tag2' or tags: tag1, tag2
            r"^\[?\"?\'?(.*?)\'?\"?\]?$",  # ["tag1", "tag2"] or 'tag1', 'tag2' or tag1, tag2 (行頭から)
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                potential_tags_str = match.group(1).strip()
                # クォートや括弧が残っている可能性があるので除去
                potential_tags_str = re.sub(r'^["\'\[\]\s]+|["\'\[\]\s]+$', "", potential_tags_str)
                # カンマで分割
                tags = [tag.strip() for tag in potential_tags_str.split(",") if tag.strip()]
                if tags:
                    logger.debug(
                        f"正規表現 ({pattern}) でカンマ区切りテキストから {len(tags)} 個のタグを抽出しました。"
                    )
                    return tags

        logger.warning(f"どの形式でもタグを抽出できませんでした。テキスト: {text[:100]}...")
        return []

    def _generate_tags(self, formatted_output) -> list[str]:
        """フォーマット済み出力からタグを生成する"""
        # デバッグ出力
        logger.debug(f"[DEBUG _generate_tags] type(formatted_output): {type(formatted_output)}")
        logger.debug(f"[DEBUG _generate_tags] formatted_output: {formatted_output}")

        # FormattedOutput（pydantic）かdictかで分岐
        if hasattr(formatted_output, "error"):
            error = formatted_output.error
            annotation = formatted_output.annotation
        else:
            error = formatted_output.get("error")
            annotation = formatted_output.get("annotation")
        logger.debug(f"[DEBUG _generate_tags] (attr) error: {error}, annotation: {annotation}")

        if error or annotation is None:
            return []

        # pydanticモデル（AnnotationSchema）かdictかで分岐
        if hasattr(annotation, "tags"):
            tags = annotation.tags
        elif isinstance(annotation, dict) and "tags" in annotation:
            tags = annotation["tags"]
        else:
            tags = None

        logger.debug(f"[DEBUG _generate_tags] tags: {tags}")

        if isinstance(tags, list):
            return tags
        return []

    @override
    def _format_predictions(self, raw_outputs: list[RawOutput]) -> list[WebApiFormattedOutput]:
        """Web API からの応答 (RawOutput) を共通の WebApiFormattedOutput にフォーマットする"""
        formatted_outputs: list[WebApiFormattedOutput] = []
        for output in raw_outputs:
            error = output.get("error")
            response_val = output.get("response")

            if error:
                formatted_outputs.append(WebApiFormattedOutput(annotation=None, error=error))
                continue

            if isinstance(response_val, AnnotationSchema):
                # AnnotationSchema型ならmodel_dump()でdictに変換
                formatted_outputs.append(
                    WebApiFormattedOutput(annotation=response_val.model_dump(), error=None)
                )
            else:
                # response_valがNoneの場合や、予期せぬ型の場合
                error_message = (
                    f"Invalid response type: {type(response_val)}"
                    if response_val is not None
                    else "Response is None"
                )
                formatted_outputs.append(WebApiFormattedOutput(annotation=None, error=error_message))
        return formatted_outputs
