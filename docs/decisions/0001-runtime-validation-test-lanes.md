# ADR 0001: Runtime Validation Test Lanes

- **日付**: 2026-05-17
- **ステータス**: Accepted

## Context

LoRAIro 側 Issue NEXTAltair/LoRAIro#276 / ADR 0026 で、LoRAIro の deterministic E2E は `ServiceContainer.annotator_library` 境界を fake に差し替え、image-annotator-lib の内部挙動を検証対象にしない方針になった。

この責任分離により、image-annotator-lib は以下の runtime contract を自リポジトリ側で検証する必要がある:

- public API (`image_annotator_lib.annotate`) から annotation runner までの契約
- model registry / `AnnotatorInfo` / capability metadata の整合
- Web API provider 呼び出し境界
- API key / provider config / retry / timeout / refusal handling
- Hugging Face 等からの local model download
- torch / transformers / local model inference
- model loading / cache / device selection / memory handling

一方で、実 API key や実 model download を mandatory PR CI に含めると、外部 provider 障害、rate limit、課金、secret 管理、model availability、network bandwidth、CPU-only runner の実行時間に test result が依存する。

したがって、通常 CI で保証する deterministic test lane と、開発者が必要時に実行する runtime validation lane を分離する。

## Decision

image-annotator-lib のテストを以下の lane に分ける。

| Lane | Scope | Trigger | Marker |
|---|---|---|---|
| Standard CI | unit / mocked integration / import smoke | PR / push | `not real_api and not heavy and not system_integration` |
| Real API validation | 実 provider API key を使う provider contract | local / manual | `real_api` |
| Local model validation | model download / 実 local inference | local / manual | `heavy` |
| System integration validation | public API から実 runtime 境界まで通す検証 | local / manual | `system_integration` |

通常 CI は Python 3.12 で固定し、以下を必須とする:

```bash
uv sync --dev
uv run pytest -m "not real_api and not heavy and not system_integration"
```

通常 CI には `import image_annotator_lib` の smoke test を含める。この smoke test は torch / torchvision 等の heavy native dependency が import 時に eager load されないことを検証する。

実 API / 実 local model / system integration validation は mandatory PR CI や scheduled CI には含めない。開発者が必要時にローカルで実行する on-demand validation として整備する。GitHub Actions の `workflow_dispatch` による manual workflow は将来追加してよいが、opt-in とし、必要な secrets / model cache / network が無い場合は skip する。

## Marker Policy

| Marker | 意味 |
|---|---|
| `real_api` | 実 provider API key を使う。通常 CI では除外する。 |
| `heavy` | model download、実 local inference、重い native dependency 初期化を伴う。通常 CI では除外する。 |
| `system_integration` | public API から provider/model runtime 境界まで通す。通常 CI では除外する。 |
| `unit` / `fast` / `standard` | 外部 runtime 依存なし。通常 CI 対象。 |
| `integration` / `fast_integration` | 外部 download/API を mock した結合検証。通常 CI 対象。 |

実 API key 不足、network 不可、model cache 不足、GPU 不在などの環境不足は、通常 CI では発生しないよう marker で除外する。on-demand validation では、環境不足を product failure と混同しないよう明示的な skip message を出す。

## Responsibility Boundary with LoRAIro

LoRAIro 側 deterministic E2E は image-annotator-lib の内部挙動を検証しない。LoRAIro は `ServiceContainer.annotator_library` 境界を fake 化し、CLI / GUI workflow、DB 保存、export など LoRAIro 管理下の配線を検証する。

image-annotator-lib は、LoRAIro から呼ばれる public API と runtime contract を自リポジトリ側で検証する。

責務分担:

| 対象 | 所在 |
|---|---|
| LoRAIro CLI / GUI workflow、DB 保存、export | LoRAIro |
| `AnnotatorLibraryAdapter` が public API に正しい引数で委譲すること | LoRAIro |
| `image_annotator_lib.annotate()` public API contract | image-annotator-lib |
| model registry / provider metadata / capability metadata | image-annotator-lib |
| Web API provider contract | image-annotator-lib |
| local model loading / inference | image-annotator-lib |

## Rationale

### なぜ mandatory CI と runtime validation を分けるか

実 API / 実 local model validation は重要だが、通常 CI の品質ゲートに含めるには不安定要素が多い。

- provider 障害や rate limit で failure になる
- API 利用料金が CI trigger に紐づく
- API key / secret 管理負担が増える
- provider response schema / model availability が予告なく変わる
- model download が大きく、CI 時間と cache 容量を消費する
- CPU-only runner では推論時間が読みにくい
- torch / transformers / CUDA 周辺の環境差が failure 原因になりやすい

mandatory CI では deterministic な contract を確認し、実 runtime 依存は必要時に明示実行する。

### なぜ LoRAIro ADR ではなく本 ADR で定義するか

LoRAIro は image-annotator-lib を外部 annotation backend として扱う。実 provider API、model loading、local inference の詳細は image-annotator-lib の責務であり、LoRAIro ADR が詳細を規定すると責任境界が曖昧になる。

本 ADR は、image-annotator-lib 側の test lane / marker / skip 条件 / CI 方針を定義し、LoRAIro 側 ADR は fake boundary のみを定義する。

### 却下した選択肢

| 案 | 却下理由 |
|---|---|
| PR ごとに実 API / 実 local model を実行する | secret / 課金 / 外部障害 / 実行時間が PR gate に混入する |
| LoRAIro 側 E2E で image-annotator-lib の runtime contract まで検証する | LoRAIro の責務を超え、外部依存でアプリ側 CI が不安定化する |
| runtime validation を手順化せず都度手動確認にする | リリース前・障害再現時の確認内容が再現不能になる |
| marker を使わず test path だけで分離する | 実行レーンの意図が pytest selection に現れず、CI 除外条件も曖昧になる |

## Consequences

### 良い点

- PR CI は deterministic かつ高速に保てる。
- 実 API / 実 model download による flake、課金、secret 管理リスクを通常 CI から除外できる。
- LoRAIro と image-annotator-lib のテスト責務境界が明確になる。
- runtime validation は必要時に同じ marker / command で再現できる。
- `import image_annotator_lib` の eager heavy import regression を通常 CI で検出できる。

### 悪い点・トレードオフ

- 実 provider / 実 local model の regression は PR CI では自動検出されない。
- リリース前や provider/model 周辺変更時に on-demand validation を実行する運用 discipline が必要になる。
- runtime validation の実行可否は開発者ローカル環境、API key、network、model cache に依存する。
- manual workflow を追加する場合は、secret skip、課金、失敗時 triage の設計が別途必要になる。

### 運用ルール

- 通常 CI は `-m "not real_api and not heavy and not system_integration"` を使う。
- 実 provider API を呼ぶテストには `real_api` を付ける。
- model download / 実 inference / heavy native dependency 初期化を伴うテストには `heavy` を付ける。
- public API から runtime 境界まで通す検証には `system_integration` を付ける。
- on-demand validation で環境不足がある場合は、明確な skip reason を出す。
- 通常 CI に runtime validation を昇格する場合は、本 ADR を見直す。

## Related

- **LoRAIro umbrella**: NEXTAltair/LoRAIro#276
- **LoRAIro ADR**: NEXTAltair/LoRAIro `docs/decisions/0026-on-demand-runtime-validation-strategy.md`
- **Issue**: NEXTAltair/image-annotator-lib#65
- **関連ファイル**:
  - `pyproject.toml` (`[tool.pytest.ini_options].markers`)
  - `.github/workflows/*`
  - `tests/unit/`
  - `tests/integration/`
