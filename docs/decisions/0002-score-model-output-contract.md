# ADR 0002: Score Model Output Contract

- **日付**: 2026-05-17
- **ステータス**: Accepted

## Context

LoRAIro 側から移管された NEXTAltair/image-annotator-lib#66 で、score 系モデルの
`UnifiedAnnotationResult` contract 不整合が見つかった。

`cafe_aesthetic` など一部 scorer は `capabilities=["scores"]` を宣言している一方で、
`UnifiedAnnotationResult.tags` に score-derived tag を設定していた。そのため、
capability validation により `tags provided but TAGS not in capabilities` の
`ValidationError` が発生する。

調査対象には 2 種類の reference がある。

1. **実装リファレンス**: `toshiaki1729/dataset-tag-editor-standalone`
2. **モデル配布元リファレンス**: Hugging Face / GitHub 上の実モデル・重み配布元

`dataset-tag-editor-standalone` は dataset caption editor の custom tagger として aesthetic
score predictor を実装している。ここでは UI の tagger API に合わせるため、score を
`[CAFE]score_6` や `very aesthetic` のようなタグ文字列に変換して返している。

一方で、image-annotator-lib の `UnifiedAnnotationResult` は capability と field の整合を
明示する contract である。scorer が score-derived tag を返すなら `TAGS` capability が
必要になるが、そうすると scorer が tagger としても扱われ、下流の検索・保存・export の
意味が曖昧になる。

### Code inventory at decision time

予備調査 (Issue #66 comment 参照) で確認した、対象 5 モデルの現状コード位置と振る舞いの divergence:

| 系統 | 対象 | `_format_predictions` の `tags` field | dead/orphan code |
|---|---|---|---|
| Pipeline | `AestheticShadow{,V2}` (`model_class/pipeline_scorers.py`) | `_generate_tags()` で `["very aesthetic"]` 等を返し contract 違反 | なし (実コード経路上の違反) |
| Pipeline | `CafePredictor` (同上) | `_generate_tags()` で `["[CAFE]_score_N"]` を返し contract 違反 | なし (実コード経路上の違反) |
| CLIP | `ImprovedAesthetic` / `WaifuAesthetic` (`model_class/scorer_clip.py`) | `core/base/clip.py:_format_predictions` で既に `tags=None` (contract 整合) | `ClipBaseAnnotator._get_score_tag` (abstractmethod) と override 2 件 + `ClipBaseAnnotator._generate_tags` が呼び出し経路ゼロの orphan |

つまり `UnifiedAnnotationResult` validation の実際の breakage 源は Pipeline 系 2 class のみで、CLIP 系は既に contract 整合済みだが score-derived tag 生成メソッドが dead code として残存している。`config/.../annotator_config.toml` の `capabilities = ["scores"]` は対象 5 モデル全てで既に正しく宣言されている。

## References

### Implementation reference

| Model | dataset-tag-editor-standalone reference | Output style |
|---|---|---|
| `aesthetic_shadow_v2` | `userscripts/taggers/aesthetic_shadow.py` | `hq` score を閾値で `very aesthetic` / `aesthetic` / `displeasing` / `very displeasing` に変換 |
| `cafe_aesthetic` | `userscripts/taggers/cafeai_aesthetic_classifier.py` | `aesthetic` probability を `[CAFE]score_N` に変換 |
| `ImprovedAesthetic` | `userscripts/taggers/improved_aesthetic_predictor.py` | CLIP+MLP prediction を `[IAP]score_N` に変換 |
| `WaifuAesthetic` | `userscripts/taggers/waifu_aesthetic_classifier.py` | CLIP+MLP prediction を `[WD]score_N` に変換 |

This repository is a useful behavior reference, but it is not the model distribution source.
Its `Tagger` API returns tags, so its score-to-tag conversion must not be copied as the
default `UnifiedAnnotationResult` scorer contract.

### Model distribution references

| image-annotator-lib model | Distribution source | Notes |
|---|---|---|
| `aesthetic_shadow_v1` | `https://huggingface.co/shadowlilac/aesthetic-shadow` | Hugging Face image-classification model. Model card describes anime image quality assessment and low-quality dataset filtering use. |
| `aesthetic_shadow_v2` | `https://huggingface.co/NEXTAltair/cache_aestheic-shadow-v2` | Mirror of `shadowlilac/aesthetic-shadow-v2`; score interpretation is `very aesthetic >= 0.71`, `aesthetic 0.45-0.71`, `displeasing 0.27-0.45`, `very displeasing <= 0.27`. |
| `cafe_aesthetic` | `https://huggingface.co/cafeai/cafe_aesthetic` | Hugging Face image-classification model fine-tuned on `microsoft/beit-base-patch16-384`; labels include `aesthetic` / `not_aesthetic`. |
| `ImprovedAesthetic` | `https://github.com/christophschuhmann/improved-aesthetic-predictor` | CLIP+MLP aesthetic score predictor. image-annotator-lib uses `sac+logos+ava1-l14-linearMSE.pth` with `openai/clip-vit-large-patch14`. |
| `WaifuAesthetic` | `https://huggingface.co/hakurei/waifu-diffusion-v1-4/tree/main/models` and `https://github.com/waifu-diffusion/aesthetic` | `aes-B32-v0.pth` with `openai/clip-vit-base-patch32`. The GitHub reference describes a 0 to 1 anime-style aesthetic score. |

## Decision

image-annotator-lib の scorer contract は **scores only** とする。

`type="scorer"` / `capabilities=["scores"]` のモデルは、`UnifiedAnnotationResult` で以下を
満たさなければならない。

- `capabilities` は `TaskCapability.SCORES` を含む。
- `scores` は model-specific な `dict[str, float]` として返す。
- `tags` は `None` にする。
- `captions` は `None` にする。
- `ratings` は scorer の正式出力に含めない。
- `raw_output` には必要に応じて元ラベル、元スコア、変換前 tensor shape などを保持してよい。

score-derived tag は `UnifiedAnnotationResult.tags` に入れない。

必要であれば、score-derived tag は別機能として明示的に設計する。候補は以下のどちらかとする。

- caller 側が `scores` を読んで tag に変換する utility
- `capabilities=["scores", "tags"]` を明示する別 model / wrapper

ただし、この ADR では scorer の標準 contract には含めない。

### Model-specific score keys

初期実装では、score key は配布元・実装元の意味を保つ。

| Model | `scores` keys | Scale |
|---|---|---|
| `aesthetic_shadow_v1` | `hq`, `lq` | pipeline の分類 probability |
| `aesthetic_shadow_v2` | `hq`, `lq` | pipeline の分類 probability |
| `cafe_aesthetic` | `aesthetic` | pipeline の `aesthetic` probability |
| `ImprovedAesthetic` | `aesthetic` | CLIP+MLP raw prediction; expected 0-10 系 |
| `WaifuAesthetic` | `aesthetic` | CLIP+MLP raw prediction; expected 0-1 系 |

score scale の正規化はこの ADR では行わない。異なる model の score を同一尺度として扱う
必要が出た場合は、別 ADR で正規化 contract を定義する。

## Rationale

### なぜ score-derived tag を正式出力にしないか

`score` は数値評価であり、`tag` は検索・分類・caption 編集用のラベルである。
dataset-tag-editor-standalone が tag を返すのは、そのアプリの custom tagger API に合わせる
ためであり、モデル配布元の primary output が tag であることを意味しない。

scorer が暗黙に tag も返すと、以下が曖昧になる。

- `capabilities=["scores"]` と `tags` field の整合
- model registry 上の scorer / tagger の区別
- LoRAIro 側 DB で通常 tag と score-derived tag を同列に保存するか
- tag 検索や export に score-derived tag を混ぜるか
- model ごとに違う prefix や閾値を library contract として固定するか

scorer は `scores` のみ返し、tag 化が必要な利用者が明示的に変換する方が責任境界が明確である。

### なぜ model-specific key / scale を残すか

対象モデルの戻り値は統一されていない。

- `cafe_aesthetic` は `aesthetic` / `not_aesthetic` の image-classification probability を返す。
- `aesthetic_shadow` は `hq` / `lq` の probability を返す。
- `ImprovedAesthetic` は CLIP embedding に MLP をかけた aesthetic prediction を返す。
- `WaifuAesthetic` は anime-styled image 向けの 0 to 1 score を返す。

これらを無理に単一 key / 単一 scale に正規化すると、モデル固有の意味を失う。まずは
配布元の意味を保った `scores: dict[str, float]` として返し、正規化が必要になった時点で
別設計にする。

### 却下した選択肢

| 案 | 却下理由 |
|---|---|
| scorer の capability に `tags` を追加する | scorer が tagger として扱われ、下流の検索・保存・export contract が曖昧になる |
| dataset-tag-editor-standalone の tag 変換をそのまま正式仕様にする | 同リポジトリは実装リファレンスであり、モデル配布元ではない。Tagger API に合わせた変換を library 標準 contract にすべきではない |
| `UnifiedAnnotationResult` 内で score-to-tag 変換を共通実装する | score scale と tag naming が model-specific で、共通化すると余計な policy を core type に持ち込む |
| 全 scorer の key を `aesthetic` に統一する | `aesthetic_shadow` の `hq` / `lq` のような配布元 label の意味を失う |

## Consequences

### 良い点

- `UnifiedAnnotationResult` の capability validation と scorer 実装が一致する。
- `cafe_aesthetic` / `aesthetic_shadow_v1` / `aesthetic_shadow_v2` の ValidationError を解消できる。
- CLIP 系 scorer の現行 `scores` only 実装と整合する。
- LoRAIro 側で score-derived tag を通常 tag と混同しない。
- dataset-tag-editor-standalone の実装を参考にしつつ、モデル配布元と library contract を分離できる。

### 悪い点・トレードオフ

- 既存の `[CAFE]score_N` / `[IAP]score_N` / `[WD]score_N` のような tag 表現は scorer の標準出力から消える。
- score-derived tag を利用したい caller には、別途 utility / wrapper が必要になる。
- LoRAIro など consumer 側は、model-specific score key と scale を意識する必要がある。

### 運用ルール

- scorer 実装の unit test は `tags is None` と `scores is not None` を明示的に検証する。
- `capabilities=["scores"]` の model が `tags` / `captions` を返した場合は regression とする。
- score-derived tag を追加する場合は、この ADR を更新するか、新しい ADR で wrapper / utility の責務を定義する。その際 `WaifuAesthetic` の現行 orphan prefix `[WAIFU]` と reference (`dataset-tag-editor-standalone`) の `[WD]` の命名乖離を再評価する (本 ADR の implementation reference table は reference の `[WD]` を採用済み)。
- 実 local model validation は ADR 0001 の `heavy` / `system_integration` lane に従う。

## Related

- **Issue**: NEXTAltair/image-annotator-lib#66
- **Preliminary investigation (code-level)**: NEXTAltair/image-annotator-lib#66 issue comment (reproduction + 行番号付き code inventory + 完了条件)
- **Runtime validation ADR**: 0001
- **LoRAIro ADR**: NEXTAltair/LoRAIro `docs/decisions/0026-on-demand-runtime-validation-strategy.md`
- **Implementation reference**: `https://github.com/toshiaki1729/dataset-tag-editor-standalone`
- **Model sources**:
  - `https://huggingface.co/shadowlilac/aesthetic-shadow`
  - `https://huggingface.co/NEXTAltair/cache_aestheic-shadow-v2`
  - `https://huggingface.co/cafeai/cafe_aesthetic`
  - `https://github.com/christophschuhmann/improved-aesthetic-predictor`
  - `https://huggingface.co/hakurei/waifu-diffusion-v1-4/tree/main/models`
  - `https://github.com/waifu-diffusion/aesthetic`
