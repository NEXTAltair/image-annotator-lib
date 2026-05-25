# ADR 0005: Provider Batch API Contract

- **日付**: 2026-05-25
- **ステータス**: Accepted

## Context

LoRAIro は Provider Batch API を direct provider route の非同期 job pipeline として採用する。
LoRAIro 側 ADR 0038 は、LoRAIro の責務を job persistence、UI、`custom_id` と DB image identity の
対応、既存 annotation save path への投入に限定した。

Provider ごとの batch API は input / submit / status / result 取得が異なる。

- OpenAI: request JSONL file を upload し、`/v1/batches` job を retrieve して output/error file を取得する
- Anthropic: Messages request array を batch create し、results endpoint から JSONL stream を読む
- Google: Gemini Developer API Batch の input file を upload し、job result file を取得する

Vertex AI BatchPredictionJob は MVP 対象外である。GCS / BigQuery、IAM、project / region、
service account 管理は LoRAIro の API key based annotation 経路と責務が違い、MVP の provider
batch abstraction に持ち込まない。

LoRAIro に provider 固有 payload や result artifact の差分を漏らすと、同期 annotation と batch
annotation の保存経路が分岐し、LoRAIro が provider parser を抱えることになる。image-annotator-lib は
既に provider/model 解釈、API key 注入、multimodal payload、通常 annotation result contract を持つため、
Provider Batch API の provider 境界も library 側に置く。

## Decision

image-annotator-lib は Provider Batch API の provider-specific lifecycle を所有し、LoRAIro には
job 単位の安定 DTO だけを公開する。

### Public API

```python
def list_batch_capable_models() -> list[BatchModelInfo]: ...

def submit_batch(request: BatchSubmitRequest) -> BatchSubmitResult: ...

def retrieve_batch(handle: BatchJobHandle) -> BatchStatusResult: ...

def cancel_batch(handle: BatchJobHandle) -> BatchStatusResult: ...

def fetch_batch_results(handle: BatchJobHandle) -> BatchFetchResult: ...
```

`list_batch_capable_models()` は引数で endpoint / prompt profile を絞り込まない。library が現在
batch 実行可能な direct-provider model をすべて返す。consumer は自分の local availability gate と
突き合わせて表示する。

```python
class BatchModelInfo:
    provider: str
    litellm_model_id: str
    display_name: str
    capabilities: set[str]
```

`BatchModelInfo` は表示と submit input の選択に必要な最小情報だけを持つ。provider-specific pricing
metadata や deprecation metadata は stable contract に含めない。

### Model eligibility

Batch-capable model は以下を満たす model とする。

- direct provider route である
- annotation に必要な vision / tool calling / structured output 相当の capability を満たす
- LiteLLM 同梱 DB に batch pricing field がある
  - `input_cost_per_token_batches`
  - `output_cost_per_token_batches`
- OpenAI `gpt-5.5-pro` family denylist に含まれない

`cache_read_input_token_cost_batches` は補助 metadata であり、単独では batch 対応判定に使わない。
OpenRouter route は対象外である。

### Provider transport

Provider Batch API の transport は公式 SDK を優先する。

- OpenAI: `openai` SDK
- Anthropic: `anthropic` SDK
- Google: `google-genai` SDK

SDK object / SDK response type は public API に露出しない。SDK を使うのは provider endpoint 呼び出し、
file upload / download、results stream 取得の薄い client layer までである。DTO、status 正規化、
error 分類、annotation result normalization は library の責務とする。

Direct HTTP 実装は MVP 方針ではない。将来、公式 SDK が対象 batch API を提供しない、または実装上
明確な障害になる場合だけ adapter 内部実装として検討する。

### Package layout

Provider Batch API の新規実装は `image_annotator_lib/webapi/batch/` に置く。

```text
image_annotator_lib/webapi/
  batch/
    __init__.py
    types.py
    service.py
    preparation.py
    adapters/
      openai.py
      anthropic.py
      google.py
```

Batch は model implementation ではないため `model_class/` には置かない。また、既存の `core/` は WebAPI
execution concern が肥大化しているため、新規 batch 実装では `core/provider_batch/` を作らない。

既存の `core/webapi_annotator.py`、`core/provider_manager.py`、`core/model_id.py`、
`core/api_model_discovery.py`、`core/result_adapter.py`、`core/output_normalization.py`、
`core/image_preprocess.py`、`core/http_retry.py` を `image_annotator_lib/webapi/` へ移す作業は別 issue とする。
MVP batch 実装では既存 import path を無理に移動せず、新規 batch code だけを `webapi/batch/` に置く。

### Submit DTO

```python
class BatchSubmitRequest:
    provider: str
    endpoint: str
    litellm_model_id: str
    prompt_profile: str
    description: str | None
    api_keys: dict[str, str]
    items: list[BatchSubmitItem]

class BatchSubmitItem:
    custom_id: str
    image_id: int
    image_path: Path
```

`custom_id` は LoRAIro が生成する `img-{image_id}` 形式を想定する。library は `custom_id` を opaque な
per-request identity として provider payload に載せ、結果でも同じ値を返す。

`image_path` は consumer が管理する送信用 resized image path である。library は provider-specific
image payload preparation を担当する。

- file existence / readability validation
- MIME / format detection
- supported image MIME validation
- provider-specific artifact / request construction 時の binary read
- provider-specific base64 / data URL construction
- provider-specific image input fields
- provider image detail / fidelity defaults

MVP では image/jpeg、image/png、image/webp を受け付ける。format conversion は行わない。
LoRAIro は resized image として WebP を渡す想定である。unsupported format は
`BatchJobError(phase=PREPARE, code="unsupported_image_format")` として submit 前 validation で
fail するが、MIME 判定や画像読み込みの内部詳細を LoRAIro に渡す責務は持たない。

Provider-specific payload の最終形状への変換は common preparer ではなく provider adapter が行う。
common preparer は次の中間表現までを作る。

```python
class PreparedBatchItem:
    custom_id: str
    image_id: int
    image_path: Path
    image_mime_type: str
    prompt: str
```

OpenAI / Google adapter は JSONL artifact を書き出す loop 内で `image_path` を読み base64 化する。
Anthropic adapter は `requests` array 構築時に `image_path` を読み base64 化し、serialized body size
を検証する。

MVP では既存の長辺 512px resized image を前提にし、`detail: high` / original fidelity の
consumer-facing option は提供しない。

```text
TODO: Consider provider image detail/fidelity options later. MVP uses consumer-provided resized images.
```

### Provider submit forms

MVP の provider submit form は以下に固定する。

- OpenAI: `/v1/responses` request JSONL を File API に upload し、`/v1/batches` を作成する
- Anthropic: Message Batches create に `requests` array を直接渡す
- Google: Gemini Developer API Batch の input file(JSONL + File API)方式を使う

Google inline requests は実装しない。LoRAIro は画像 annotation で最大 500 items を扱うため、
inline request body size 制限と相性が悪い。input file 方式の方が OpenAI と同じ JSONL artifact
処理に寄せやすく、大きい batch に適する。

Provider 共通で 1 batch 最大 500 items とする。この制限は LoRAIro のコスト事故防止と、
Anthropic Message Batches の request body size 制約を踏まえた conservative guard である。
Anthropic adapter は serialized request body が 256 MB を超える場合、submit 前に
`BatchJobError(phase=PREPARE, code="payload_too_large")` とする。

### Submit / handle / status DTO

```python
class BatchSubmitResult:
    provider: str
    provider_job_id: str
    status: BatchStatus
    request_count: int

class BatchJobHandle:
    provider: str
    provider_job_id: str
    api_keys: dict[str, str]

class BatchStatusResult:
    provider: str
    provider_job_id: str
    status: BatchStatus
    request_count: int | None
    succeeded_count: int | None
    failed_count: int | None
    canceled_count: int | None
    expired_count: int | None
    submitted_at: datetime | None
    completed_at: datetime | None
    expires_at: datetime | None

class BatchStatus(StrEnum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
```

`provider_job_id` は provider が返す batch job id であり、consumer が作る任意 ID ではない。
OpenAI `output_file_id` などの provider-specific auxiliary id は stable contract に含めない。
必要な補助 ID は `provider_job_id` から retrieve して library 内部で扱う。

API key は通常 annotation と同じ `api_keys` dict を各 API 呼び出しで受け取る。MVP では provider client
cache は持たず、呼び出しごとに client を構築する。

Provider native status は `BatchStatus` に丸める。OpenAI の `validating` / `in_progress` /
`finalizing` など、LoRAIro が個別分岐しない詳細状態は `RUNNING` に寄せる。LoRAIro にとって重要な
状態は、待つ、結果取得可能、失敗、cancel、expire、unknown の区別である。

### Result DTO

```python
class BatchFetchResult:
    provider: str
    provider_job_id: str
    status: BatchStatus
    items: list[BatchResultItem]

class BatchResultItem:
    custom_id: str
    status: BatchItemStatus
    provider_status: BatchProviderItemStatus
    annotation: UnifiedAnnotationResult | None
    error: BatchItemError | None

class BatchItemStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class BatchProviderItemStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"

class BatchItemError:
    phase: BatchErrorPhase
    code: str
    message: str
    retryable: bool

class BatchErrorPhase(StrEnum):
    PREPARE = "prepare"
    UPLOAD = "upload"
    SUBMIT = "submit"
    POLL = "poll"
    CANCEL = "cancel"
    DOWNLOAD = "download"
    PARSE = "parse"
    NORMALIZE = "normalize"
    VALIDATE = "validate"
```

`BatchFetchResult` は job 単位で返す。`BatchResultItem.annotation` は同期 annotation と同じ
`UnifiedAnnotationResult` contract に正規化する。consumer は provider-native response body を parse しない。

`BatchResultItem.status` は annotation outcome を表す。provider item が HTTP/API 上成功していても、
annotation output として正規化できなければ `FAILED` とする。provider-native success/failure は
`provider_status` に分離する。

Provider-specific result retrieval と normalization は library が担当する。

- OpenAI file download / JSONL parse
- Anthropic results stream / JSONL parse
- Google result file download / JSONL parse
- provider error body の `BatchItemError` への正規化

MVP では streaming iterator API は提供しない。result file / stream を library 内部で処理し、
`BatchFetchResult.items` として返す。大規模 batch の memory pressure が実問題になった場合だけ、
後続 ADR / issue で iterator API を検討する。

Result artifact の長期保存機能は提供しない。library は provider result artifact を取得・parse・正規化し、
`BatchFetchResult` だけを返す。provider raw response 全文や raw request payload を LoRAIro に渡さない。
Debug が必要な場合は library-side log で扱う。

### Error handling

Batch error は job-level と item-level に分ける。

```python
class BatchJobError(Exception):
    phase: BatchErrorPhase
    provider: str
    provider_job_id: str | None
    code: str
    message: str
    retryable: bool
```

Job-level error は provider job operation 自体が成立しない、または job artifact 全体を処理できない失敗である。
`submit_batch()` / `retrieve_batch()` / `cancel_batch()` / `fetch_batch_results()` は該当時に
`BatchJobError` を raise し、失敗 result DTO は返さない。

代表例:

- `PREPARE`: `too_many_items`, `unsupported_image_format`, `missing_api_key`, `payload_too_large`
- `UPLOAD`: OpenAI / Google input JSONL upload failure
- `SUBMIT`: provider batch job create failure
- `POLL`: status retrieve failure
- `CANCEL`: cancel failure
- `DOWNLOAD`: job not completed, result artifact download failure
- `PARSE`: result JSONL 全体が読めない

`fetch_batch_results()` が未完了 job に呼ばれた場合は
`BatchJobError(phase=DOWNLOAD, code="job_not_completed", retryable=True)` とする。

Item-level error は job artifact は読めるが、特定 item を annotation として使えない失敗である。
代表 code は以下とする。

- `provider_item_error`
- `safety_refusal`
- `content_policy_refusal`
- `image_policy_violation`
- `invalid_image`
- `max_tokens`
- `annotation_output_unparseable`
- `annotation_schema_invalid`
- `missing_result`
- `unknown_item_error`

Provider item success は annotation success と同義ではない。HTTP 2xx / provider item error null でも、
OpenAI `message.refusal` / Responses `content.type == "refusal"`、Anthropic `stop_reason == "refusal"`、
Google `finishReason == "SAFETY"` などの native signal があれば item failure とする。Native signal が
無い拒否文や自由文応答は refusal と断定せず、annotation output として parse / validation できなければ
`annotation_output_unparseable` または `annotation_schema_invalid` とする。

Refusal/rating/error record としてどう保存するか、今後その画像を送信対象から外すかは LoRAIro の責務である。
library は raw provider response ではなく、`BatchItemError` の `phase` / `code` / `message` /
`retryable` を返す。

### Retry policy

library は API 呼び出し単位の一時エラーだけを軽く retry する。job 単位の自動再 submit と item 単位の
自動再 submit は行わない。

Retry 対象:

- timeout
- connection error
- 429 / rate limit
- 5xx
- SDK が retryable と分類する一時エラー

Retry しない:

- request validation error
- invalid API key
- unsupported model
- payload too large
- bad request
- safety / content policy refusal
- output parse / schema validation failure

`retryable` は LoRAIro / user initiated retry の判断材料であり、library が batch job を自動で作り直す
ことを意味しない。

### Annotation outcome normalization

通常 annotation と batch annotation は ADR 0006 の annotation outcome classifier を使う。実行経路は
違うが、結果の意味は同じである。

```text
provider response / exception
  -> provider-specific outcome signal normalization
  -> annotation output extraction
  -> AnnotationSchema / UnifiedAnnotationResult validation
```

通常 annotation は既存互換のため `UnifiedAnnotationResult.error` の文字列 prefix contract を維持してよい。
Batch API は構造化された `BatchItemError` を返す。分類ロジックは ADR 0006 で共通化し、外側の戻し方だけを
分ける。

Batch で native safety / refusal signal を検出した場合、LoRAIro 側は通常 annotation と同じ意味論で
`SafetyRefusalError` / `ContentPolicyRefusalError` 相当へマップできる。文字列 prefix 実装は batch DTO には
持ち込まない。

Prompt profile と output schema は通常 annotation と同じものを使う。Batch 専用 prompt / validation を
作ると通常 annotation と結果差分が出て保存経路も分岐するため、MVP では採用しない。

### Stable contract に含めないもの

以下は stable LoRAIro/library contract に含めない。

- provider raw payload / raw response JSON
- provider-specific handles (`output_file_id`, `error_file_id`, result file name など)
- persisted provider result artifact
- token / cost estimate
- post-completion usage / cost summary
- retention timeline
- expected completion timestamp
- OpenRouter batch route
- Vertex AI BatchPredictionJob / GCS / BigQuery based batch route
- Google inline requests route
- `gpt-5.5-pro` family の batch eligibility

Provider raw data が必要になった場合は、library-side debug logging として別途扱う。

## Rationale

### なぜ provider-specific 処理を library に置くか

Provider Batch API は provider ごとに submit 形式、status、artifact 取得方法が違う。consumer が
それらを直接扱うと、provider ごとの parser と validation が LoRAIro 側に広がる。

library 側に閉じれば、LoRAIro は同期 annotation と同じ annotation result contract を受け取れる。
これにより DB 保存、tag / caption / rating / score 保存、error handling は既存経路に寄せられる。

### なぜ job 単位で返すか

LoRAIro は result item だけでなく、job status と request counts も同時に更新する。
そのため `fetch_batch_results()` は item iterator ではなく、job-level metadata と item list を含む
`BatchFetchResult` を返す。

### なぜ raw provider payload を返さないか

Provider raw payload は debug には便利だが、stable contract に含めると provider schema drift が
consumer contract に漏れる。library が validation と normalization を担当するため、LoRAIro が raw
provider JSON を保存・解釈する必要はない。

### なぜ `provider_job_id` だけを stable handle にするか

Batch status / cancel / fetch は provider job id から retrieve できる。OpenAI の output file id などは
retrieve 後に得られる補助 ID であり、stable consumer contract に含める必要はない。Google / Anthropic
への拡張でも、provider job id 相当の handle に寄せる方が API が薄く保てる。

### なぜ artifact 保存を持たないか

Provider result JSONL を保存すると、後で再 parse できる利点はある。しかし raw provider response の長期保存、
データ肥大化、privacy 管理、provider ごとの差分管理が発生する。MVP では library が fetch 時に parse し、
正規化済み result だけを返す。LoRAIro の DB 保存処理が失敗した場合の再投入機能は別設計で扱う。

## Consequences

### 良い点

- LoRAIro は provider-specific batch result format を知らずに済む
- 通常 annotation と同じ result contract を再利用できる
- API key の渡し方は既存 annotation と同じ `api_keys` dict に揃う
- OpenAI / Anthropic / Google の result retrieval 差分を library 内に閉じられる
- cost / retention / completion prediction を MVP に持ち込まず、実装範囲を抑えられる
- 通常 annotation と batch annotation の outcome 分類を揃えられる

### 悪い点・トレードオフ

- library が batch lifecycle と provider SDK/API 差分を追加で背負う
- LoRAIro 側で provider raw response を直接 inspect できない
- provider result artifact を後で再 parse する機能は MVP では持たない
- 大規模 batch では `BatchFetchResult.items` の list 化が memory pressure になる可能性がある
- `gpt-5.5-pro` family denylist は hard-coded cost-safety rule であり、将来見直しが必要になる可能性がある

## Related

- LoRAIro ADR 0038: Provider Batch API Integration Strategy
- image-annotator-lib ADR 0002: Score Model Output Contract
- image-annotator-lib ADR 0003: Rating Model Output Contract
- image-annotator-lib ADR 0006: Annotation Outcome Contract
