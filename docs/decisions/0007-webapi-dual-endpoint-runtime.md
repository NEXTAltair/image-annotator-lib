# ADR 0007: WebAPI Dual-Endpoint Runtime (Chat + Responses)

- **日付**: 2026-06-01
- **ステータス**: Accepted

> **Design Authority:**
> WebAPI 推論経路の責務分離 (PydanticAI / LiteLLM / image-annotator-lib) の上位方針は
> LoRAIro 側 [ADR 0023 — PydanticAI / LiteLLM WebAPI Inference Boundary](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md)
> が SSoT。本 ADR は ADR 0023 の責務境界を前提に、**runtime での endpoint 選択
> (Chat Completions / Responses) の具体 contract** を image-annotator-lib 側に記録する。
> 上位方針は LoRAIro ADR 0023 の `Amendment (2026-06-01, iam-lib #131)` を参照。

## Context

OpenAI は Responses API を主軸 endpoint に位置づけ、pro ティアのモデル
(`gpt-5-pro`, `o3-pro`, `o1-pro`, `gpt-5.x-pro` 系) は `mode=responses` 専用で
Chat Completions (`/v1/chat/completions`) には載っていない。

iam-lib の WebAPI runtime (`webapi/provider_manager.py` + `webapi/model_id.py`) は
OpenAI provider を無条件に Chat Completions 経路 (`OpenAIChatModel` / `v1/chat/completions`)
で構築していた。このため responses 専用モデルを指定すると実行時に provider 側で
404 になっていた。

#130 (merged, PR #132) では暫定処置として discovery 段階の endpoint-gate を
`mode=chat` のみに絞り、responses 系モデルを registry から除外した。これは 404 を
止めるための応急対応であり、pro ティアは「実行可能だが除外されている」状態だった。

#131 では gate を反転し、`OpenAIResponsesModel` 経路を runtime に実装して
pro ティアを実行可能にする。これに合わせて、annotation に不適なモデル群
(codex / deep-research) の除外責務を endpoint 互換性の判定から分離して整理する。

litellm 同梱 DB 実測 (2026-06-01):

- gate 反転で復活する openai モデル: pro ティア 11 件
  (`gpt-5-pro` + dated, `gpt-5.2/5.4/5.5-pro` + dated, `o1-pro` + dated, `o3-pro` + dated)。
  全て `vision=True` / `function_calling=True`。
- 除外維持: deep-research 系 4 件 (data-source tool 前提)。
- 新規除外: codex 系 (openai-direct 8 件 + `openrouter/openai/gpt-5.1-codex-max`)。

## Decision

WebAPI 同期 runtime の OpenAI 経路を、**litellm `mode` 由来の per-model endpoint 選択**
に拡張する。

### 軸の分離

endpoint 選択 (実行可否) と annotation 適性 (使うべきか) を**コード上 2 軸に分離**する。

| 軸 | 意味 | #131 での扱い |
|---|---|---|
| 軸A: endpoint 互換性 | provider 上でそのモデルがどの endpoint で実行可能か | discovery gate を反転し `chat` + `responses` の両方を許可 |
| 軸B: annotation 適性 | 画像 annotation 用途に向くか | denylist を維持・拡張 (codex 一律除外 / deep-research 除外維持) |

軸A の gate と軸B の denylist は別ロジックとして実装し、片方の変更が他方に
波及しないようにする。

### endpoint 選択経路 (per-model)

endpoint 選択は litellm registry metadata の `mode` field を source とし、
以下の経路で per-model に配線する。

```text
litellm DB mode field
  -> discovery / registry metadata (mode 格納)
  -> annotator
  -> provider_manager
  -> resolve_model_ref(litellm_model_id, ..., mode=...)
  -> PydanticAIModelRef.endpoint
  -> build_pydantic_model(ref, ...) 内の endpoint 分岐
```

分岐規則:

- provider が `openai` **かつ** `mode == "responses"` のとき → `OpenAIResponsesModel`
- 上記以外の openai → `OpenAIChatModel`
- anthropic / google / openrouter → 常に Chat 系 (endpoint 分岐なし)

`OpenAIResponsesModel` 分岐は openai-direct provider に限定する。OpenRouter 経由の
openai モデル (`openrouter/openai/...`) は OpenRouter が Chat 互換 endpoint を提供する
ため Chat 経路のまま扱う。

### codex 一律除外 (軸B / denylist)

codex 系モデルは provider / endpoint を問わず **一律除外**する。

- openai-direct の responses codex
- 現在 chat で有効だった `openrouter/openai/gpt-5.1-codex-max`

も含め、denylist substring `codex` で除外する。

これは plan_130 の「動作可能なため openrouter codex は残す」という当初判断を
**#131 で上書き**したものである (経緯は Rationale 参照)。

### deep-research 除外維持 (軸B / denylist)

deep-research 系モデルは data-source tool を前提とするため、gate 反転後も
denylist で除外を維持する。endpoint が responses として許可されても annotation
には使わない。

### batch は chat 専用のまま (ADR 0038 不変)

LoRAIro ADR 0038 (Provider Batch API Integration Strategy) が OpenAI batch 経路で
`/v1/chat/completions` を選択する判断は **変更しない**。#131 は同期 (sync) runtime
のみを対象とし、batch 経路の endpoint 選択には触れない。

理由は LoRAIro Issue #518 の lesson と整合する: batch を chat に固定することで
sync と batch の endpoint / tool schema / response shape を一致させ、fixture と
refusal 分類ロジックを両経路で再利用できる。responses 専用 pro ティアは sync
runtime でのみ実行可能になる。

### refusal 契約は不変

iam-lib [ADR 0006 (Annotation Outcome Contract)](0006-annotation-outcome-contract.md)
および LoRAIro ADR 0023 Phase 1.5 の refusal 契約 (exception `body` 再帰 walk +
type 名 + regex signature による provider 横断分類) は Responses 経路でもそのまま
機能する想定。本 ADR で refusal 関連のコード変更は行わず、実 API smoke で確認する
(LoRAIro [ADR 0026](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0026-on-demand-runtime-validation-strategy.md))。

## Rationale

### なぜ per-model endpoint 選択か

OpenAI の endpoint は固定値ではなくモデルごとに異なる (一部は responses 専用、
多くは chat)。provider 単位で endpoint を固定すると、片方の endpoint 専用モデルが
必ず実行不能になる。litellm 同梱 DB は `mode` field を SSoT として持つため、
これを per-model に配線すれば「どの endpoint で構築すべきか」を registry metadata
だけで決定でき、推論層に provider 別の特例分岐を増やさずに済む。

これは PydanticAI 2.0 beta の「Responses 強制全切替」を採らない plan_525 の判断とも
**矛盾しない**。#131 は全切替ではなく、litellm `mode` 由来で chat / responses を
per-model に選ぶだけであり、chat モデルは引き続き `OpenAIChatModel` で実行される。

### なぜ codex を一律除外するか (plan_130 の上書き)

plan_130 では「openrouter codex は chat で動作可能なため残す」としていた。これは
軸A (実行可否) のみを見た判断だった。#131 で軸A / 軸B を分離した結果、codex は
コーディング特化モデルであり画像 annotation には不適という軸B の観点が明確になった。
そのため「動作可能か」ではなく「annotation に使うべきか」を基準に、provider /
endpoint を問わず substring `codex` で一律除外する方針へ変更した。

### 軸分離の利点

- 軸A (endpoint gate) の変更は「実行可能なモデル集合」だけを動かし、annotation
  適性の判断には影響しない。
- 軸B (denylist) の変更は「使うべきモデル集合」だけを動かし、endpoint 構築ロジック
  には影響しない。
- 新しい OpenAI モード / 新しい不適モデルカテゴリが出ても、片方の軸の編集で対応でき、
  もう片方の回帰を起こしにくい。

## Consequences

### 良い点

- responses 専用の pro ティア (`gpt-5-pro`, `o3-pro`, `o1-pro`, `gpt-5.x-pro` 系
  11 件) が runtime で実行可能になる。
- endpoint 構築が litellm metadata の `mode` 由来で per-model に決まり、推論層に
  provider 別の endpoint 特例分岐を増やさない。
- annotation 不適モデル (codex / deep-research) の除外が軸B denylist に集約され、
  endpoint 互換性判定 (軸A) から独立する。

### トレードオフ

- Responses 経路の tool 公開形式 / response shape は Chat 経路と完全一致しない
  可能性があり、structured output と refusal 分類が pro ティアで期待通りに動くかは
  実 API smoke での確認が必要 (ADR 0026)。
- denylist は litellm 同梱 DB の更新でモデルが増減すると drift しうる。codex /
  deep-research の新モデル名が substring 規則から漏れる / 過剰一致する可能性がある
  ため、月次 dependency review (litellm bump 時) で denylist の妥当性を確認する。
- batch は chat 専用のまま据え置くため、responses 専用 pro ティアは batch 経路では
  利用できない (sync runtime 限定)。

## Related

- LoRAIro [ADR 0023 — PydanticAI / LiteLLM WebAPI Inference Boundary](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md) (`Amendment (2026-06-01, iam-lib #131)` が上位方針の SSoT)
- LoRAIro [ADR 0038 — Provider Batch API Integration Strategy](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0038-provider-batch-api-integration-strategy.md) (batch は chat 専用のまま不変)
- LoRAIro [ADR 0026 — On-Demand Runtime Validation Strategy](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0026-on-demand-runtime-validation-strategy.md) (responses 経路の実 API smoke 方針)
- iam-lib [ADR 0006 — Annotation Outcome Contract](0006-annotation-outcome-contract.md) (refusal 分類契約は不変)
- image-annotator-lib #131 (gate 反転 + `OpenAIResponsesModel` 経路実装) / #130 (PR #132, 暫定 chat-only gate)
- LoRAIro #589
- plan_130 (openrouter codex を残す当初判断 — 本 ADR で上書き) / plan_518 (batch を chat 固定する lesson) / plan_525 (Responses 強制全切替を不採用)
