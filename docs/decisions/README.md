# Architecture Decision Records

image-annotator-lib の重要な設計判断を記録するドキュメント群。

| ADR | タイトル | 日付 | ステータス |
|-----|---------|------|-----------|
| [0001](0001-runtime-validation-test-lanes.md) | Runtime Validation Test Lanes | 2026-05-17 | Accepted |
| [0002](0002-score-model-output-contract.md) | Score Model Output Contract | 2026-05-17 | Accepted |
| [0003](0003-rating-model-output-contract.md) | Rating Model Output Contract | 2026-05-21 | Accepted |
| [0004](0004-onnx-tagger-loader-and-model-selection.md) | ONNX Tagger Loader and Model Selection | 2026-05-21 | Accepted |

## ADR テンプレート

```markdown
# ADR XXXX: タイトル

- **日付**: YYYY-MM-DD
- **ステータス**: Proposed | Accepted | Deprecated | Superseded by [XXXX]

## Context

なぜこの決定が必要だったか。問題の背景と制約。

## Decision

何を決定したか。

## Rationale

なぜこの選択をしたか。他の選択肢との比較。

## Consequences

この決定による影響。良い点・悪い点・トレードオフ。
```
