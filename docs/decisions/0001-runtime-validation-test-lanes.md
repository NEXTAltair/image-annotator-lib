# ADR 0001: Runtime Validation Test Lanes

- **日付**: 2026-05-17
- **Amended**: 2026-05-18
- **ステータス**: Accepted (amended)

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

image-annotator-lib のテストを以下の lane に分ける (ADR amendment 2026-05-18 で 3 lane → 2 lane に統合)。

| Lane | Scope | Trigger | Marker |
|---|---|---|---|
| Standard CI | unit / mocked integration / import smoke | PR / push | `not downloads_and_runs_model and not calls_real_webapi` |
| Real WebAPI validation | 実 provider API key で実リクエストを送信 | **local only** | `calls_real_webapi` |
| Local model validation | model download + 実 local inference | **local only** | `downloads_and_runs_model` |

旧 `system_integration` lane は廃止。public API から実 runtime 境界までの検証は、他 2 lane の test 内容で吸収する (`tests/runtime_validation/` 配下の test は `image_annotator_lib.annotate()` public API を必ず経由する)。

通常 CI は Python 3.12 で固定し、以下を必須とする:

```bash
uv sync --dev
uv run pytest -m "not downloads_and_runs_model and not calls_real_webapi"
```

通常 CI には `import image_annotator_lib` の smoke test を含める。この smoke test は torch / torchvision 等の heavy native dependency が import 時に eager load されないことを検証する。

実 WebAPI / 実 local model validation は CI に一切含めない (mandatory / scheduled / `workflow_dispatch` すべて禁止、ADR amendment 2026-05-18)。開発者が必要時にローカルで `tests/runtime_validation/` 配下の test を手動実行する。再導入する場合は本 ADR と LoRAIro ADR 0026 の両方を破棄し、新規 ADR で再設計する。

## Marker Policy

| Marker | 意味 |
|---|---|
| `calls_real_webapi` | 実 provider API key で実リクエストを送信する on-demand validation。**CI 不経由、ローカル only**。 |
| `downloads_and_runs_model` | model download + 実 local inference + 重い native dependency 初期化を伴う on-demand validation。**CI 不経由、ローカル only**。 |
| `unit` / `fast` / `standard` | 外部 runtime 依存なし。通常 CI 対象。 |
| `integration` / `fast_integration` | 外部 download/API を mock した結合検証。通常 CI 対象。 |

旧 marker (`real_api` / `heavy` / `system_integration`) は ADR amendment 2026-05-18 で `calls_real_webapi` / `downloads_and_runs_model` の 2 軸に統合済。`system_integration` lane 自体は廃止。

実 API key 不足、network 不可、model cache 不足、GPU 不在などの環境不足は、通常 CI では発生しないよう marker で除外する。on-demand validation では、環境不足を product failure と混同しないよう明示的な skip message を出す (Skip 仕様は下記参照)。

## `tests/runtime_validation/` ディレクトリの責務範囲

実 model DL + 実推論を伴う test、および実 WebAPI key で実リクエストを送る test は `tests/runtime_validation/` 配下に集約する。

**置くべき test**:
- 実 provider API key で実リクエストを送信 (`@pytest.mark.calls_real_webapi`)
- Hugging Face 等から実 model を DL し実推論 (`@pytest.mark.downloads_and_runs_model`)
- `image_annotator_lib.annotate()` public API を経由した実 runtime 通過検証 (旧 `system_integration` lane の責務を吸収)

**置くべきでない test**:
- mock / patch を使う test (`tests/unit/` または `tests/integration/` に配置)
- fake backend で動く E2E (LoRAIro 側 ADR 0026 の責務)
- 軽量 import smoke test (`tests/unit/` に配置)

## Marker 組み合わせ運用

`calls_real_webapi` と `downloads_and_runs_model` は排他ではない。LoRAIro adapter 経由で実 model と実 WebAPI を同 test で検証する場合、両 marker を併用する。

```python
@pytest.mark.downloads_and_runs_model
@pytest.mark.calls_real_webapi
def test_adapter_with_real_model_and_webapi(...): ...
```

filter 上は `not (downloads_and_runs_model or calls_real_webapi)` で CI から完全除外される。

## Skip 仕様

`tests/runtime_validation/` 配下の test は、環境不足を product failure と区別するために明示的に skip する:

- 実 API key が `os.environ` に無い場合: `pytest.skip(f"{API_KEY_NAME} not set")`
- GPU 不在 (CUDA 非対応環境) で GPU 必須 model を実行する場合: `pytest.skip("CUDA not available")`
- network 不可 (model DL 失敗) の場合: `pytest.skip("network unavailable")`
- model cache 不足 (DL 完了前) の場合: 自動 DL 実施 (skip しない、初回のみ時間がかかる旨を log で明示)

skip 時は理由を明示し、product failure と混同させない。

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

- 通常 CI は `-m "not downloads_and_runs_model and not calls_real_webapi"` を使う。
- 実 provider API key で実リクエストを送るテストには `calls_real_webapi` を付け、`tests/runtime_validation/` に配置する。
- model download / 実推論 / heavy native dependency 初期化を伴うテストには `downloads_and_runs_model` を付け、`tests/runtime_validation/` に配置する。
- 旧 `system_integration` lane (public API → 実 runtime 通過検証) は廃止。同等の検証は他 2 lane の test 内容で吸収する (test は public API `annotate()` を必ず経由)。
- on-demand validation で環境不足 (API key 未設定 / GPU 不在 / network 不可) がある場合は、明確な skip reason を出す (Skip 仕様セクション参照)。
- 実 API / 実モデル validation を CI (scheduled / `workflow_dispatch` / mandatory) に昇格することは禁止 (本 ADR amendment 2026-05-18)。再導入する場合は本 ADR と LoRAIro ADR 0026 の両方を破棄し、新規 ADR で再設計する。

## Related

- **LoRAIro umbrella**: NEXTAltair/LoRAIro#276
- **LoRAIro ADR**: NEXTAltair/LoRAIro `docs/decisions/0026-on-demand-runtime-validation-strategy.md` (amended 2026-05-18)
- **Issue**: NEXTAltair/image-annotator-lib#65
- **Amendment tracking Issue**: NEXTAltair/image-annotator-lib#71
- **関連ファイル**:
  - `pyproject.toml` (`[tool.pytest.ini_options].markers`)
  - `.github/workflows/*`
  - `tests/unit/`
  - `tests/integration/`
  - `tests/runtime_validation/` (amendment 2026-05-18 で SSoT 化)
