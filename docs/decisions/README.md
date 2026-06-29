---
type: Reference
title: Architecture Decision Records
status: Accepted
timestamp: 2026-06-29
tags: [process]
---
# Architecture Decision Records

image-annotator-lib の重要な設計判断を記録するドキュメント群。各 ADR は先頭に YAML
frontmatter（`type` / `title` / `status` / `timestamp` / 任意 `tags` / `depends_on`）を持つ。
docs 全体の frontmatter 規約は [ADR 0010](0010-okf-frontmatter-for-docs.md) を参照。

<!-- OKF-TABLE:START -->
| ADR | タイトル | 日付 | ステータス |
|---|---|---|---|
| [0001](0001-runtime-validation-test-lanes.md) | Runtime Validation Test Lanes | 2026-05-17 | Accepted (amended 2026-05-18) |
| [0002](0002-score-model-output-contract.md) | Score Model Output Contract | 2026-05-17 | Accepted |
| [0003](0003-rating-model-output-contract.md) | Rating Model Output Contract | 2026-05-21 | Accepted |
| [0004](0004-onnx-tagger-loader-and-model-selection.md) | ONNX Tagger Loader and Model Selection | 2026-05-21 | Accepted |
| [0005](0005-provider-batch-api-contract.md) | Provider Batch API Contract | 2026-05-25 | Accepted |
| [0006](0006-annotation-outcome-contract.md) | Annotation Outcome Contract | 2026-05-25 | Accepted |
| [0007](0007-webapi-dual-endpoint-runtime.md) | WebAPI Dual-Endpoint Runtime (Chat + Responses) | 2026-06-01 | Accepted |
| [0008](0008-webapi-bounded-image-concurrency.md) | WebAPI Bounded Image Concurrency | 2026-06-03 | Accepted |
| [0009](0009-scorer-value-range-reference.md) | Scorer Raw Output and Value-Range Reference | 2026-06-05 | Accepted |
| [0010](0010-okf-frontmatter-for-docs.md) | OKF YAML Frontmatter for Documentation | 2026-06-29 | Accepted |
<!-- OKF-TABLE:END -->

> このテーブルは `make adr-index` が frontmatter から生成する。手編集しない。

## ADR テンプレート

```markdown
---
type: ADR
title: "タイトル"
status: Proposed | Accepted | Deprecated | Superseded
timestamp: YYYY-MM-DD
tags: []
depends_on: []
---
# ADR XXXX: タイトル

## Context

なぜこの決定が必要だったか。問題の背景と制約。

## Decision

何を決定したか。

## Rationale

なぜこの選択をしたか。他の選択肢との比較。

## Consequences

この決定による影響。良い点・悪い点・トレードオフ。
```
