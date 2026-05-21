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

`UnifiedAnnotationResult` validation の breakage 源は Pipeline 系 2 class が `tags` field に
categorical label / 整数 bin tag を入れている点。本 ADR の Decision に従い、

- Pipeline 系 (canonical label を持つ): `tags=...` を `score_labels=...` に rename
  (`AestheticShadow` の閾値マッチ logic は維持、`CafePredictor` は argmax label 化に変更)
- CLIP 系 (regression のみ): orphan の `_get_score_tag` / `_generate_tags` を削除
- `CafePredictor._generate_tags` (整数 bin tag): 削除 (lib では生成しない)
- `config/.../annotator_config.toml`: 該当 3 モデルの `capabilities` を
  `["scores", "score_labels"]` に更新

config TOML の `capabilities = ["scores"]` は 5 モデル全てで宣言済みだが、3 モデルは
`SCORE_LABELS` 追加が必要。

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

image-annotator-lib の scorer contract は、**配布元が canonical な categorical label を
提供するモデル**では `["scores", "score_labels"]`、純 regression モデルでは `["scores"]`
とする。

新規 capability `TaskCapability.SCORE_LABELS` と新規 field `score_labels: list[str] | None`
を `UnifiedAnnotationResult` に追加し、score から導出される **categorical 分類ラベル**を
`tags` field とは独立に保持する。

### 共通ルール (全 scorer)

`type="scorer"` のモデルは、`UnifiedAnnotationResult` で以下を満たさなければならない。

- `capabilities` に `TaskCapability.SCORES` を含む。
- `scores` は model-specific な `dict[str, float]` として返す。
- `tags` は `None` にする。
- `captions` は `None` にする。
- `raw_output` には必要に応じて元ラベル、元スコア、変換前 tensor shape などを保持してよい。

### Canonical label を持つ scorer (`["scores", "score_labels"]`)

配布元 (model card / `config.json` 等) で categorical 分類ラベルが明示されているモデルでは、
`SCORE_LABELS` capability も持ち、`score_labels: list[str]` に label を返す。

該当モデル: `aesthetic_shadow_v1`, `aesthetic_shadow_v2`, `cafe_aesthetic`

`score_labels` の生成ルール:

- `aesthetic_shadow_v1/v2`: model card 記載の 4-tier 閾値 (`very aesthetic >= 0.71`,
  `aesthetic 0.45-0.71`, `displeasing 0.27-0.45`, `very displeasing <= 0.27`) を `hq`
  probability に適用し、該当 tier 1 つを `["very aesthetic"]` 形式で返す。
- `cafe_aesthetic`: 2-class image-classification の argmax label
  (`["aesthetic"]` or `["not_aesthetic"]`) を返す。

整数 bin tag (`[CAFE]score_N` / `[IAP]score_N` / `[WD]score_N`) は **生成しない**
(reference 実装 `toshiaki1729/dataset-tag-editor-standalone` の UI tagger API 専用変換で
あり、配布元保証ではないため)。

### 純 regression scorer (`["scores"]`)

配布元が categorical label を明示しないモデル (連続値 regression のみ) は、`SCORES`
capability のみで `score_labels` は `None` とする。

該当モデル: `ImprovedAesthetic` (1-10 連続値), `WaifuAesthetic` (0-1 連続値)

label 化が必要な consumer は、raw `scores` を読んで閾値マッチを自前実装する。

### `tags` field を使わない理由

`UnifiedAnnotationResult.tags` は content tag (WDTagger / DeepDanbooru 等の画像内容
記述) 用途。scorer の `very aesthetic` / `aesthetic` のような categorical label を `tags`
に入れると、下流 DB / 検索 / export で content tag と区別不能になる。専用 field
`score_labels` で分離する。

### Model-specific score keys

初期実装では、score key は配布元・実装元の意味を保つ。

| Model | `capabilities` | `scores` keys | Scale | `score_labels` 例 | Canonical source for key/scale |
|---|---|---|---|---|---|
| `aesthetic_shadow_v1` | `["scores", "score_labels"]` | `hq`, `lq` | 0-1 binary classification probability (`hq + lq = 1`) | `["very aesthetic"]` / `["aesthetic"]` / `["displeasing"]` / `["very displeasing"]` (v2 の閾値を慣習的に流用) | `config.json` の `id2label = {"0": "hq", "1": "lq"}` (model card 本文には未記載)。閾値は v2 model card 由来の慣習適用 |
| `aesthetic_shadow_v2` | `["scores", "score_labels"]` | `hq`, `lq` | 0-1 binary classification probability (`hq + lq = 1`) | 同上 (canonical from model card) | `config.json` の `id2label = {"0": "hq", "1": "lq"}` + model card の 4-tier 閾値 (`very aesthetic >= 0.71` 等) |
| `cafe_aesthetic` | `["scores", "score_labels"]` | `aesthetic`, `not_aesthetic` | 0-1 image-classification probability (両 label 共に保持) | `["aesthetic"]` or `["not_aesthetic"]` (argmax) | Model card label list (`aesthetic` / `not_aesthetic`) |
| `ImprovedAesthetic` | `["scores"]` | `aesthetic` | **1-10 系** CLIP+MLP linearMSE 回帰出力 | (N/A: regression) | LAION-Aesthetics 公式 blog: 訓練データ (SAC / LAION-Logos / AVA) が `1 to 10` MOS rating であり、公開 subset の閾値も `>= 4.5 / 5 / 6` で記述されている |
| `WaifuAesthetic` | `["scores"]` | `aesthetic` | **0-1 系** CLIP+MLP 回帰出力 | (N/A: regression) | `waifu-diffusion/aesthetic` README: `"a 0 to 1 score, where the lowest score means... low aesthetic rating and a high score means... high aesthetic rating"` (verbatim) |

score scale の正規化はこの ADR では行わない。異なる model の score を同一尺度として扱う
必要が出た場合は、別 ADR で正規化 contract を定義する。

### Canonical source 確認の重要性

score key 名や scale は、必ずしも model card 本文に書かれていない。例えば `aesthetic_shadow_v1/v2`
の `hq` / `lq` ラベルは model card には記載がなく、`config.json` の `id2label` で初めて
正式定義されていることを確認できる。同様に `ImprovedAesthetic` の 1-10 scale は GitHub README
には書かれておらず、LAION 公式 blog の訓練データ説明と公開 subset 閾値 (`>= 5` 等) からのみ
確認できる。

scorer 追加時 / 既存 scorer の挙動を疑う際は、以下の優先順位で canonical source を確認する:

1. `config.json` の `id2label` / `label2id` (HuggingFace image-classification model)
2. Model card 本文の labels / threshold 記述
3. 公式 blog / 訓練データセットの scoring scale 記述
4. 配布元 README の usage 例 (`Prediction: 0.999...` のような出力例)

Model card 単独では不足する場合があるため、`config.json` と訓練データ仕様を併読する。

## Rationale

### なぜ canonical label を持つ scorer だけ `score_labels` を返すか

整形ルール (閾値マッチ / argmax) の出所が**配布元保証**であるモデルでは、lib が label を
返しても arbitrary policy を持ち込まない。具体的には:

- `aesthetic_shadow_v2`: model card 本文に 4-tier 閾値が verbatim 記載 (canonical)
- `cafe_aesthetic`: model card の labels (`aesthetic` / `not_aesthetic`) は config.json
  の `id2label` でも定義されている canonical な 2-class label
- `aesthetic_shadow_v1`: v2 と同 architecture (`config.json:id2label = {"0": "hq", "1": "lq"}`
  共通)、閾値は v2 から慣習的に流用 (strictly canonical ではないが reference 実装と一致)

一方、`ImprovedAesthetic` / `WaifuAesthetic` の整数 bin tag (`[IAP]score_N` /
`[WD]score_N`) は dataset-tag-editor-standalone の UI tagger API 専用変換で、配布元
(`christophschuhmann/improved-aesthetic-predictor` README / `waifu-diffusion/aesthetic`
README) はラベル化ルールを示していない。これらの整数 bin を lib contract に固定すると
arbitrary policy になるため、`SCORE_LABELS` を持たせず regression のみとする。

### なぜ `tags` field を再利用しないか

`UnifiedAnnotationResult.tags` は content tag (画像内容を記述するラベル: WDTagger の
`1girl` / `outdoor` 等) 用途で確立済み。scorer の categorical label を `tags` に入れると
下流で:

- DB 上で content tag と score categorical label が同じテーブルに混在
- tag 検索 / export filter が両者を区別できない
- consumer の用途 (内容検索 vs 品質フィルタ) で意図しないマッチを発生させる

別 field (`score_labels`) で分離することで、capability validation と downstream の
処理を明示的に分ける。

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
| scorer の categorical label を `tags` field に格納する | `tags` は content tag 用途で確立済。混在すると下流 DB / 検索 / export contract が曖昧になる |
| dataset-tag-editor-standalone の整数 bin tag 変換 (`[CAFE]score_N` / `[IAP]score_N` / `[WD]score_N`) をそのまま正式仕様にする | 同リポジトリは UI tagger API 用 implementation reference であり、モデル配布元ではない。配布元保証のない arbitrary policy を library 標準 contract にすべきではない |
| `UnifiedAnnotationResult` 内で score-to-tag 変換を共通実装する | score scale と tag naming が model-specific で、共通化すると余計な policy を core type に持ち込む |
| 全 scorer の key を `aesthetic` に統一する | `aesthetic_shadow` の `hq` / `lq` のような配布元 label の意味を失う |
| 全 scorer に `score_labels` を強制する | regression モデル (`ImprovedAesthetic` / `WaifuAesthetic`) は配布元が categorical label を提供しない。lib 側で閾値を arbitrary に決めることになる |
| `score_labels` ではなく `ratings` という field 名を使う | `ratings` は別概念 (例: NSFW rating 等) と混同しやすい。本 ADR は scorer の categorical label に特化した名前 `score_labels` を採用 |

ADR 0003 で `ratings` field を rating / NSFW classifier 用に定義した。本 ADR で却下したのは
score-derived categorical label を `ratings` と呼ぶ案であり、rating model output contract とは
矛盾しない。

## Consequences

### 良い点

- `UnifiedAnnotationResult` の capability validation と scorer 実装が一致する。
- `cafe_aesthetic` / `aesthetic_shadow_v1` / `aesthetic_shadow_v2` の ValidationError を解消できる。
- canonical label を持つモデルでは lib が直接 label を返すため consumer 側で再実装不要。
- content tag (WDTagger 等) と score categorical label が field レベルで分離され、下流 DB / 検索 / export で区別可能。
- 整数 bin tag (`[CAFE]score_N` 等) のような配布元保証のない arbitrary policy を library に持ち込まない。

### 悪い点・トレードオフ

- `UnifiedAnnotationResult` schema 変更 (`SCORE_LABELS` capability + `score_labels` field 追加)。downstream consumer は新 field を認識する必要がある。
- `[CAFE]score_N` / `[IAP]score_N` / `[WD]score_N` の整数 bin tag 表現は lib の標準出力には含まれない。必要な consumer は raw `scores` から自前で整数 bin 化する。
- regression scorer (`ImprovedAesthetic` / `WaifuAesthetic`) で categorical label が必要な場合は consumer 側で閾値マッチ実装が必要。
- `aesthetic_shadow_v1` の閾値は model card に明示なし (v2 から慣習流用)、strictly canonical ではない。

### 運用ルール

- 純 regression scorer (`["scores"]`) の unit test は `tags is None` / `score_labels is None` / `scores is not None` を明示的に検証する。
- canonical label scorer (`["scores", "score_labels"]`) の unit test は `tags is None` / `score_labels is not None` / `scores is not None` を明示的に検証する。
- `capabilities=["scores"]` の model が `tags` / `captions` / `score_labels` を返した場合は regression とする。
- `tags` field に score categorical label を入れた場合は regression とする (downstream で content tag と区別不能)。
- 整数 bin tag 表現 (`[CAFE]score_N` 等) を consumer 側で生成する場合、reference 実装 (`toshiaki1729/dataset-tag-editor-standalone`) の整形ルールを記録 (本 ADR の Implementation reference table + Issue #66 comment)。`WaifuAesthetic` の現行 lib 内 orphan `_get_score_tag` は scale 変換抜けの bug があるため踏襲しないこと (Issue #66 comment 参照)。
- 実 local model validation は ADR 0001 の `heavy` / `system_integration` lane に従う。

## Related

- **Issue**: NEXTAltair/image-annotator-lib#66
- **Preliminary investigation (code-level)**: NEXTAltair/image-annotator-lib#66 issue comment (reproduction + 行番号付き code inventory + 完了条件)
- **Runtime validation ADR**: 0001
- **Rating output ADR**: 0003
- **LoRAIro ADR**: NEXTAltair/LoRAIro `docs/decisions/0026-on-demand-runtime-validation-strategy.md`
- **Implementation reference**: `https://github.com/toshiaki1729/dataset-tag-editor-standalone`
- **Model sources**:
  - `https://huggingface.co/shadowlilac/aesthetic-shadow`
  - `https://huggingface.co/NEXTAltair/cache_aestheic-shadow-v2`
  - `https://huggingface.co/cafeai/cafe_aesthetic`
  - `https://github.com/christophschuhmann/improved-aesthetic-predictor`
  - `https://huggingface.co/hakurei/waifu-diffusion-v1-4/tree/main/models`
  - `https://github.com/waifu-diffusion/aesthetic`
