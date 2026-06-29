---
type: ADR
title: "OKF YAML Frontmatter for Documentation"
status: Accepted
timestamp: 2026-06-29
deciders: NEXTAltair
tags: [process, config]
depends_on: [yaml]
---
# ADR 0010: OKF YAML Frontmatter for Documentation

## Context

ADR `0001–0009` までは frontmatter を持たず、メタデータ（日付・ステータス）を本文の
インライン bullet（`- **日付**` / `- **ステータス**`）で表現していた。フォーマットも揺れており
（`# ADR NNNN:` と `# NNNN.`、`## Status` 節と bullet が混在）、`docs/` 配下の非 ADR
ドキュメントには共通の frontmatter 規約が無かった。

下流の LoRAIro は **ADR 0069**（ADR を OKF バンドル化）と **ADR 0082**（通常ドキュメントへ
OKF frontmatter を拡張、LoRAIro#971）で、YAML frontmatter をドキュメントの SSoT として扱い、
横断検索・索引生成・エージェント参照を揃える方針を採った。姉妹パッケージの
**genai-tag-db-tools も ADR 0010 で同じ規約に追従**している。本リポジトリは LoRAIro に
消費される下流パッケージであり、3 リポジトリ横断で種別・ドメイン・依存技術を機械可読に
するため、同じ OKF frontmatter 規約へ寄せる。

参考: LoRAIro ADR 0069 / ADR 0082（LoRAIro#971）, genai-tag-db-tools ADR 0010。

## Decision

`docs/` 配下の Markdown ドキュメントは、先頭に **YAML frontmatter（`---` で囲む）** を持つ
ことを規約とする。frontmatter を唯一の正準ソース（SSoT）とし、本文にスカラー値メタデータ
（日付・ステータス）を置かない。

### フィールド

| キー | 必須 | 説明 |
|------|------|------|
| `type` | ✅ | 文書種別（下記語彙） |
| `title` | 任意 | 表示タイトル |
| `status` | 任意 | 状態（下記語彙） |
| `timestamp` | 任意 | 作成日・決定日・最終重要更新日。`YYYY-MM-DD` |
| `tags` | 任意 | 機能・責務・動作の抽象分類（技術名は入れない） |
| `depends_on` | 任意 | 強く依存する技術・ライブラリ・外部仕様 |
| `deciders` | ADR のみ | 決定者 |

- `version` は**持たない**。版・鮮度は `timestamp` と Git 履歴で扱う。
- `type` と重複する分類は `tags` に入れない。
- 内部 package 名はファイルパスで判別できるため、`packages` のような frontmatter は持たない。
- ADR の H1 は `# ADR NNNN: <タイトル>` に統一する。日付/ステータスは frontmatter に集約する。

### 語彙（最小限・拡張は本 ADR を改訂）

**`type`**: `ADR` / `Guide` / `Reference` / `Contract` / `Plan` / `Investigation` / `Report`

**`status`**: `Draft` / `Proposed` で始め、`Accepted` / `Implemented` / `Deprecated` /
`Superseded` / `Rejected` のいずれかで始める。詳細は ` (…)` で付与してよい。

**`tags`**（機能・責務・動作。技術名は禁止）:
`annotation` / `tagging` / `scoring` / `rating` / `captioning` / `model-selection` /
`model-registry` / `provider-batch` / `webapi` / `memory-management` / `validation` /
`performance` / `error-handling` / `config` / `process`

**`depends_on`**（技術・ライブラリ・外部仕様）:
`torch` / `transformers` / `onnxruntime` / `tensorflow` / `pydanticai` / `litellm` /
`openai-api` / `anthropic-api` / `google-genai` / `huggingface-hub` / `pydantic` / `yaml`

語彙は最小限から始め、必要が出たら本 ADR を改訂して追加する（語彙ドリフトを防ぐ）。

### 適用範囲

**必須対象**: `docs/**/*.md`（`docs/decisions/*.md` は全件 frontmatter 必須）。

**対象外**: `README.md` / `CHANGELOG.md` / `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` /
`index.md` / `log.md`（OKF 予約）/ 生成ドキュメント / 外部ツールが固有フォーマットを
要求するファイル（例: `SKILL.md`）。

### 移行戦略

- `docs/decisions/0001–0009` は本 ADR 導入時に frontmatter へ移行する（インライン日付/
  ステータスを frontmatter へ吸収）。
- 非 ADR の中核ドキュメント（`docs/integrations.md` 等）は eager に付与し、一過性・履歴的な
  ドキュメントは新規作成・実質更新時に lazy 付与する。

### 検証・索引生成（CI 自走しない）

LoRAIro ADR 0069 / 0082 と同様、検証は CI で強制せず**エージェントが判断で起動**する。
汎用スキル `okf-bundle`（stdlib only）の `okf_validate.py` / `okf_index.py` を vendor し、
`Makefile` の target で回す:

- `adr-okf`: `docs/decisions` を全件 frontmatter 必須で検証 + index 最新チェック。
- `adr-index`: frontmatter から `docs/decisions/README.md`（表）+ `index.md` を生成。
- `docs-okf`: `docs` を `--skip-missing` で検証（lazy migration）。

## Rationale

| 選択肢 | 概要 | 採否 |
|-------|------|------|
| A. frontmatter=SSoT + okf-bundle 検証 (本 ADR) | LoRAIro / genai-tag と同一規約に揃える | **採用** |
| B. 現状維持（インライン bullet メタ） | 規約なし・フォーマット揺れ継続 | 却下: 横断検索・索引化できない |
| C. 独自 frontmatter 規約 | iam-lib 専用キーで定義 | 却下: 3 リポジトリ横断の一貫性を失う |

- **下流契約としての一貫性**: 本リポジトリは LoRAIro に消費される。同じ OKF 規約に揃えると
  下流・エージェントが 3 リポジトリを同じ方法で索引・参照できる。
- **既存資産の再利用**: `okf-bundle` は `--bundle-root` で任意ディレクトリに使える stdlib ツール。

## Consequences

### 良い点

- ◎ ドキュメントの種別・ドメイン・依存技術が機械可読になり横断検索できる。
- ◎ `docs/decisions/README.md` / `index.md` が frontmatter からの生成物になり手動同期が不要。
- ◎ LoRAIro / genai-tag-db-tools と運用が揃う。

### トレードオフ

- △ ドキュメント新規作成・実質更新のたびに frontmatter 付与が必要（エージェントが判断）。
- △ 語彙が固定されるため、新ドメイン/新技術の登場時は本 ADR の改訂が要る。

## Related

- [Open Knowledge Format SPEC v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
- LoRAIro ADR 0069（ADR を OKF バンドル化）/ ADR 0082（LoRAIro#971）
- genai-tag-db-tools ADR 0010（OKF YAML Frontmatter for Documentation）
- Issue #158
