# ADR 0006: Annotation Outcome Contract

- **日付**: 2026-05-25
- **ステータス**: Accepted

## Context

image-annotator-lib は同期 annotation と Provider Batch API の両方で WebAPI model output を扱う。
実行経路は異なるが、consumer にとって重要な意味は同じである。

```text
provider response / exception
  -> annotation として使えるか
  -> safety / content policy refusal か
  -> image / token / provider error か
  -> parse / schema validation failure か
```

既存の同期 annotation は PydanticAI 経由で実行し、成功時は `UnifiedAnnotationResult`、失敗時は
`UnifiedAnnotationResult.error` の文字列として consumer に返している。Safety / content policy refusal
だけは `"SafetyRefusalError: ..."` / `"ContentPolicyRefusalError: ..."` の prefix を LoRAIro が decode
して `error_records` に保存する。

Provider Batch API は item ごとに provider-native response body を返す。OpenAI のように HTTP/API 上は
成功していても、`message.refusal` / Responses `content.type == "refusal"` を持つ場合や、単なる拒否文で
annotation schema に変換できない場合がある。つまり provider success と annotation success は同義ではない。

Batch だけ別の outcome 分類を持つと、同期 annotation と batch annotation で refusal / parse failure /
validation failure の扱いがずれる。これを避けるため、結果分類ロジックを共通化する。

## Decision

同期 annotation と batch annotation は、同じ internal annotation outcome classifier を使う。

```text
provider response / exception
  -> provider-specific outcome signal normalization
  -> annotation output extraction
  -> AnnotationSchema / UnifiedAnnotationResult validation
  -> AnnotationOutcome
```

外側の戻し方だけを実行経路ごとに変える。

- 同期 annotation: 既存互換のため `UnifiedAnnotationResult.error` 文字列 contract に変換する
- batch annotation: `BatchResultItem.annotation` または `BatchItemError` に変換する

### Internal DTO

```python
class AnnotationOutcome:
    status: AnnotationOutcomeStatus
    annotation: UnifiedAnnotationResult | None
    error: AnnotationError | None

class AnnotationOutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class AnnotationError:
    code: AnnotationErrorCode
    message: str
    retryable: bool
    provider: str
    litellm_model_id: str
```

`AnnotationOutcome` は library internal contract である。LoRAIro へ直接公開する stable DTO ではない。
Public surface では、同期 annotation は既存 result contract、batch annotation は ADR 0005 の
`BatchResultItem` / `BatchItemError` に写す。

### Error codes

MVP の `AnnotationErrorCode` は以下に絞る。

```python
class AnnotationErrorCode(StrEnum):
    SAFETY_REFUSAL = "safety_refusal"
    CONTENT_POLICY_REFUSAL = "content_policy_refusal"
    IMAGE_POLICY_VIOLATION = "image_policy_violation"
    INVALID_IMAGE = "invalid_image"
    MAX_TOKENS = "max_tokens"
    PROVIDER_ERROR = "provider_error"
    ANNOTATION_OUTPUT_UNPARSEABLE = "annotation_output_unparseable"
    ANNOTATION_SCHEMA_INVALID = "annotation_schema_invalid"
    UNKNOWN = "unknown"
```

Provider raw code は必要に応じて internal log に残す。LoRAIro には provider raw response body や SDK
例外の生情報を渡さない。

### Provider signal normalization

Provider-specific parser は native signal を先に見る。native signal がある場合は、自由文解析より優先する。

#### OpenAI

OpenAI は `finish_reason` 単独では refusal 判定しない。OpenAI の `finish_reason` は token generation
termination reason であり、model refusal が `finish_reason="stop"` と同時に返ることがあるためである。

OpenAI parser は以下を refusal / error signal として扱う。

- Responses API `response.error.code == "image_content_policy_violation"`
  - `IMAGE_POLICY_VIOLATION`
- Responses API image input error
  - `INVALID_IMAGE`
  - 例: `invalid_image`, `invalid_image_format`, `invalid_base64_image`, `invalid_image_url`,
    `image_too_large`, `image_too_small`, `image_parse_error`, `invalid_image_mode`,
    `image_file_too_large`, `unsupported_image_media_type`, `empty_image_file`,
    `failed_to_download_image`, `image_file_not_found`
- Responses API `incomplete_details.reason == "content_filter"`
  - `CONTENT_POLICY_REFUSAL`
- Responses API `incomplete_details.reason == "max_output_tokens"`
  - `MAX_TOKENS`
- Responses API output content item with `type == "refusal"`
  - `SAFETY_REFUSAL`
- Chat Completions `choices[].message.refusal`
  - `SAFETY_REFUSAL`
- Chat Completions `choices[].finish_reason == "content_filter"`
  - `CONTENT_POLICY_REFUSAL`
- Chat Completions `choices[].finish_reason == "length"`
  - `MAX_TOKENS`

`finish_reason == "stop"` かつ content が `"I'm sorry..."` のような拒否文である場合は、native signal が無い。
この場合、拒否とは断定せず、annotation output として parse / validation できなければ
`ANNOTATION_OUTPUT_UNPARSEABLE` または `ANNOTATION_SCHEMA_INVALID` とする。

#### Anthropic

Anthropic parser は Messages response の `stop_reason == "refusal"` を native refusal signal として扱う。

- `stop_reason == "refusal"`
  - `SAFETY_REFUSAL`

Anthropic でも model-generated refusal が通常 text response として返る場合がある。この場合は OpenAI と同じく
自然文から refusal と断定せず、annotation output として parse / validation できるかで成功判定する。

#### Google Gemini

Google parser は Gemini response の safety / block signal を native refusal signal として扱う。

- `promptFeedback.blockReason` が safety / policy block を示す
  - `SAFETY_REFUSAL`
- `candidates[].finishReason == "SAFETY"`
  - `SAFETY_REFUSAL`
- image input が provider error として返る
  - `INVALID_IMAGE` または `IMAGE_POLICY_VIOLATION`
- token 上限で output が完了しない
  - `MAX_TOKENS`

Google の `safetyRatings` は provider-specific evidence であり、MVP の public DTO には出さない。
必要な場合は library debug log に残す。

### Parse / validation rules

Native signal が無い場合、annotation output extraction と schema validation を行う。

- annotation 候補を抽出できない、空、JSON/tool/schema output ではない
  - `ANNOTATION_OUTPUT_UNPARSEABLE`
- JSON / structured output はあるが `AnnotationSchema` / `UnifiedAnnotationResult` に合わない
  - `ANNOTATION_SCHEMA_INVALID`
- `UnifiedAnnotationResult` を構築できた
  - `SUCCEEDED`

Free text refusal の文面、例えば `"I'm sorry, but I can't assist with this request."` のような自然文だけを見て
`SAFETY_REFUSAL` とは分類しない。LLM の拒否文は provider / model / version / prompt によって揺れるため、
heuristic refusal detection は MVP に含めない。

### Retryability

`retryable` は caller / user initiated retry の判断材料であり、library が自動再実行することを意味しない。

MVP の item-level retryability:

- `MAX_TOKENS`: `True`
- transport / rate limit / server error 由来の `PROVIDER_ERROR`: `True`
- `SAFETY_REFUSAL`: `False`
- `CONTENT_POLICY_REFUSAL`: `False`
- `IMAGE_POLICY_VIOLATION`: `False`
- `INVALID_IMAGE`: `False`
- `ANNOTATION_OUTPUT_UNPARSEABLE`: `False`
- `ANNOTATION_SCHEMA_INVALID`: `False`
- `UNKNOWN`: `False`

同期 annotation / batch annotation ともに、schema validation failure や parse failure に対して
job/item 単位の自動再 submit は行わない。同期 annotation の PydanticAI `output_retries` は既存 contract として
維持するが、classifier が failure と判断した後の追加 retry は行わない。

### Mapping to existing sync annotation result

同期 annotation は既存 consumer 互換のため、当面 `UnifiedAnnotationResult.error` の文字列 contract を維持する。

Mapping:

```text
SAFETY_REFUSAL
  -> "SafetyRefusalError: {message}"

CONTENT_POLICY_REFUSAL
  -> "ContentPolicyRefusalError: {message}"

other AnnotationErrorCode
  -> "{code}: {message}"
```

LoRAIro は既存通り `SafetyRefusalError:` / `ContentPolicyRefusalError:` prefix を decode できる。
その他の error は同期 annotation の既存 L1 failure と同じく、annotation result error として扱う。

### Mapping to batch result

Batch annotation は ADR 0005 の `BatchResultItem` に写す。

```text
AnnotationOutcome.SUCCEEDED
  -> BatchResultItem.status = SUCCEEDED
  -> BatchResultItem.annotation = outcome.annotation
  -> BatchResultItem.error = None

AnnotationOutcome.FAILED
  -> BatchResultItem.status = FAILED
  -> BatchResultItem.annotation = None
  -> BatchResultItem.error.phase = phase mapped from outcome.error.code
  -> BatchResultItem.error.code = outcome.error.code
  -> BatchResultItem.error.message = outcome.error.message
  -> BatchResultItem.error.retryable = outcome.error.retryable
```

Batch の provider item が HTTP/API 上成功していても、`AnnotationOutcome.FAILED` なら
`BatchResultItem.status = FAILED` とする。Provider-native success/failure は ADR 0005 の
`provider_status` に分離する。

Batch phase mapping:

```text
SAFETY_REFUSAL
CONTENT_POLICY_REFUSAL
IMAGE_POLICY_VIOLATION
INVALID_IMAGE
MAX_TOKENS
PROVIDER_ERROR
ANNOTATION_OUTPUT_UNPARSEABLE
UNKNOWN
  -> BatchErrorPhase.NORMALIZE

ANNOTATION_SCHEMA_INVALID
  -> BatchErrorPhase.VALIDATE
```

Job-level `PREPARE` / `UPLOAD` / `SUBMIT` / `POLL` / `CANCEL` / `DOWNLOAD` / `PARSE` は ADR 0005 の
`BatchJobError` 側で扱う。`AnnotationOutcome` は result item を annotation として使えるかの分類に限定する。

### Storage responsibility

library は rating refusal / blocklist / error_records の永続化判断を持たない。

- library: provider signal / parse / validation を `AnnotationOutcome` へ分類する
- LoRAIro: `AnnotationOutcome` 由来の sync error prefix または batch `BatchItemError` を見て保存先を決める

LoRAIro が safety refusal を rating / blocklist / error_records のどれに保存するかは LoRAIro 側 ADR / issue で
扱う。library は provider raw response 全文や category score を stable contract として渡さない。

## Rationale

### なぜ outcome classifier を共通化するか

同期 annotation と batch annotation は実行経路が違うだけで、結果の意味は同じである。Batch だけ独自分類を
持つと、同じ provider response が同期では `SafetyRefusalError`、batch では単なる parse failure になるような
ズレが起きる。

共通 classifier にすれば、provider-specific signal の追従、parse / validation failure の扱い、retryable 判定を
1 か所に集約できる。

### なぜ自然文 refusal を判定しないか

自然文 refusal は provider / model / prompt によって文面が揺れる。`"I'm sorry"` や `"I can't assist"` のような
heuristic に依存すると偽陽性・偽陰性が増える。MVP では native signal と schema validation を正本とし、
自然文しかない場合は annotation output failure として扱う。

### なぜ sync result の prefix contract を残すか

LoRAIro は既に `UnifiedAnnotationResult.error` の `SafetyRefusalError:` /
`ContentPolicyRefusalError:` prefix を decode して `error_records` に保存している。同期 annotation の public
contract を同時に変えると影響範囲が広い。内部 classifier を先に導入し、外側の既存互換 adapter で prefix
文字列へ変換する。

### なぜ batch は構造化 error を返すか

Batch API は新規 contract であり、既存の文字列 prefix 互換に縛られない。item ごとの `code` / `message` /
`retryable` を構造化して返すことで、LoRAIro は UI summary、保存判断、手動 retry 判断を文字列 parse なしで
行える。

## Consequences

### 良い点

- 同期 annotation と batch annotation の refusal / parse / validation semantics が揃う
- Provider-specific signal 追従を 1 か所に集約できる
- OpenAI の `finish_reason="stop"` refusal 罠を classifier 側で明示的に扱える
- Batch は構造化 error を返せるため、LoRAIro が provider raw response を parse しなくて済む
- 既存同期 annotation consumer との互換を保てる

### 悪い点・トレードオフ

- 既存同期 annotation の `error: str` contract と新規 batch の structured error が当面併存する
- `AnnotationOutcome` から sync prefix 文字列への adapter が必要になる
- Native signal が無い自然文 refusal は refusal として保存されず、parse failure として扱われる
- Provider parser が OpenAI / Anthropic / Google の response schema drift に追従する必要がある

## Related

- ADR 0005: Provider Batch API Contract
- LoRAIro ADR 0023: PydanticAI / LiteLLM WebAPI Inference Boundary
