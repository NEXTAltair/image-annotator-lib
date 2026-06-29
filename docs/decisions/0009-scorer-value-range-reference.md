---
type: ADR
title: "Scorer Raw Output and Value-Range Reference"
status: Accepted
timestamp: 2026-06-05
tags: [scoring, annotation]
depends_on: [torch]
---
# ADR 0009: Scorer Raw Output and Value-Range Reference

- **関連 ADR**: [0002 Score Model Output Contract](0002-score-model-output-contract.md)
- **関連 Issue**: NEXTAltair/LoRAIro#626 (スコア表示の尺度合わせ方針), NEXTAltair/image-annotator-lib#66 (scorer output contract), NEXTAltair/image-annotator-lib#144 (score_scales 実装)

## Context

ADR 0002 は scorer の `UnifiedAnnotationResult` contract (capability / field 整合, model-specific
score key) を定義したが、**score scale の正規化は明示的にスコープ外**とし、以下を予約している
(0002 `:145-146`):

> score scale の正規化はこの ADR では行わない。異なる model の score を同一尺度として
> 扱う必要が出た場合は、別 ADR で正規化 contract を定義する。

LoRAIro 側でスコア表示の尺度合わせ (異なる scorer の数値スコアを共通スケールで見せる /
フィルタする / 手動編集スライダーに乗せる) を設計する必要が出てきた (LoRAIro#626)。
正規化方式を決める前に、**各 scorer の「モデルの生の戻り値」「lib `_format_predictions` 変換後の値」
「配布元一次ソースで確認した値域」を一箇所に確定**しておく必要がある。

本 ADR は **その reference を確定すること自体を目的**とする。正規化 contract (どの方式で
共通スケールへ写像するか) は本 reference を土台に後続改訂 / 別 ADR で決定する
(`## Open Questions` 参照)。

Status は当初 `Proposed` だったが、Issue #144 で本 reference の値域を
`UnifiedAnnotationResult.score_scales` メタデータとして lib が宣言する実装
(`ScoreScale` dataclass + 各 scorer の値域宣言) が完了したため `Accepted` に確定する。
値域 reference (Reference 1–7) が lib の実装に裏打ちされた SSoT として固まったことをもって
確定とみなす。正規化方式そのものは依然として未決定で consumer 責務 (`## Open Questions` /
Reference 8 参照)。

### 検証方法

各値域は 2026-06-05 時点で以下を一次ソースとしてクロスチェックした:

- 配布元リポジトリ (HuggingFace model card / `config.json` / GitHub 推論コード) を実取得
- lib 実装 (`core/base/clip.py`, `core/base/pipeline.py`, `model_class/pipeline_scorers.py`,
  `model_class/scorer_clip.py`) のコード経路を確認
- system config (`resources/system/annotator_config.toml`) の `activation_type` /
  `final_activation_type` を確認

## Decision

本 ADR では **正規化方式を決定しない**。以下の reference を lib の SSoT として確定し、
正規化 contract の検討材料とする。

### Reference 1: 配布元リポジトリ (一次ソース)

| Model | class | 配布元 (一次ソース) | 生存状況 (2026-06-05) |
|---|---|---|---|
| `aesthetic_shadow_v1` | `AestheticShadow` | [shadowlilac/aesthetic-shadow](https://huggingface.co/shadowlilac/aesthetic-shadow) | 生存 (月 ~155 DL)。model card は "prediction score" 言及のみで hq/lq ラベル・閾値の記載なし |
| `aesthetic_shadow_v2` | `AestheticShadow` | 公式 `shadowlilac/aesthetic-shadow-v2` は**配布停止**、ミラー [NEXTAltair/cache_aestheic-shadow-v2](https://huggingface.co/NEXTAltair/cache_aestheic-shadow-v2) | ミラー依存。4-tier 閾値は shadowlilac model card ではなく **Animagine-XL 3.1 由来** (Reference 5 参照) |
| `cafe_aesthetic` | `CafePredictor` | [cafeai/cafe_aesthetic](https://huggingface.co/cafeai/cafe_aesthetic) | 生存 (月 ~2,496 DL)。base: `microsoft/beit-base-patch16-384` |
| `ImprovedAesthetic` | `ImprovedAesthetic` | [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) (`sac+logos+ava1-l14-linearMSE.pth`) | 生存。CLIP backbone: `openai/clip-vit-large-patch14` |
| `WaifuAesthetic` | `WaifuAesthetic` | [waifu-diffusion/aesthetic](https://github.com/waifu-diffusion/aesthetic) + 重み [hakurei/waifu-diffusion-v1-4](https://huggingface.co/hakurei/waifu-diffusion-v1-4/tree/main/models) (`aes-B32-v0.pth`) | 生存。CLIP backbone: `openai/clip-vit-base-patch32` |

> 注: model card 本文だけでは値域・ラベルが不足するケースが多い (ADR 0002 `:148-163`)。
> 例えば `aesthetic_shadow` の `hq`/`lq` は model card になく `config.json` の `id2label` が
> 唯一の canonical。`ImprovedAesthetic` の 1-10 scale は GitHub README になく LAION blog の
> 訓練データ (AVA/SAC/LAION-Logos の MOS rating) からのみ確認できる。

### Reference 2: モデルの生の戻り値 (raw model output)

| Model | framework | 生出力の構造 | 最終活性化 | 値域 | 有界性 |
|---|---|---|---|---|---|
| `aesthetic_shadow_v1/v2` | HF Pipeline (image-classification) | `[{'label':'hq','score':p}, {'label':'lq','score':1-p}]` | softmax (pipeline 内) | 各 0–1, `hq + lq = 1` | **有界** (確率) |
| `cafe_aesthetic` | HF Pipeline (image-classification) | `[{'label':'aesthetic','score':p}, {'label':'not_aesthetic','score':1-p}]` | softmax (pipeline 内) | 各 0–1, sum = 1 | **有界** (確率) |
| `ImprovedAesthetic` | CLIP+MLP (pytorch) | CLIP ViT-L/14 image_features → L2 normalize → MLP `768→1024→128→64→16→1` | **なし** (`#nn.ReLU()` で無効, 出力層も活性化なし) | 概ね 1–10 (AVA MOS 訓練), clamp なし | **非有界** |
| `WaifuAesthetic` | CLIP+MLP (pytorch) | CLIP ViT-B/32 image_features → L2 normalize → MLP `512→256→1` | **Sigmoid** (`final_activation_type = "Sigmoid"`, `annotator_config.toml:40`) | 0–1 (例: `Prediction: 0.999903...`) | **有界** |

確認済みコード経路:
- Pipeline 系: `core/base/pipeline.py:_run_inference` が HF pipeline をそのまま呼ぶ。
  softmax は pipeline 内部。lib は確率を**無加工**で受ける。
- CLIP 系: `core/base/clip.py:_run_inference (:102-123)` が
  `image_features / image_features.norm(...)` で L2 正規化後 `classifier_head(image_features)`
  を通し `.squeeze(-1)`。**clamp / 値域正規化なし**。最終活性化は config の
  `final_activation_type` で head 内に組み込まれる (Waifu=Sigmoid, Improved=なし)。

### Reference 3: lib `_format_predictions` 変換後の値

| Model | `capabilities` | `scores` (keys / 値域) | `score_labels` | 変換ロジックの場所 |
|---|---|---|---|---|
| `aesthetic_shadow_v1/v2` | `["scores","score_labels"]` | `{hq, lq}` 生確率 0–1 | hq に 4-tier 閾値 (`>=0.71 / 0.45 / 0.27`、未満は `very displeasing`) → 単一ラベル。閾値の出所は Reference 5 | `pipeline_scorers.py:46-85` |
| `cafe_aesthetic` | `["scores","score_labels"]` | `{aesthetic, not_aesthetic=1-aesthetic}` 0–1 | argmax (`aesthetic` if `>0.5` else `not_aesthetic`) → 単一ラベル | `pipeline_scorers.py:101-153` |
| `ImprovedAesthetic` | `["scores"]` | `{aesthetic: raw}` **無加工・非有界** | `None` | `core/base/clip.py:132-159` |
| `WaifuAesthetic` | `["scores"]` | `{aesthetic: raw}` **無加工・0–1** | `None` | `core/base/clip.py:132-159` |

**重要**: lib は値域の正規化・clamp・共通スケール化を **一切していない**。`scores` の数値は
配布元/重みが出した生値そのまま。CLIP 系は両モデルとも key が `aesthetic` で衝突するが
値域が異なる (Improved=非有界 ~1-10 / Waifu=0-1) 点に注意。

### Reference 4: 外部 reference 実装の変換 (採用しない既知変換)

`toshiaki1729/dataset-tag-editor-standalone` の UI tagger API は score をタグ文字列に変換する
(ADR 0002 `:55-66`)。lib はこれを採用しない (配布元保証のない arbitrary policy のため) が、
正規化方式を検討する際の既存事例として記録する:

| Model | reference 変換 | 備考 |
|---|---|---|
| `aesthetic_shadow` | hq → `very aesthetic` / `aesthetic` / `displeasing` / `very displeasing` | lib の `score_labels` と同等 |
| `cafe_aesthetic` | aesthetic prob → `[CAFE]score_N` (整数 bin) | lib は不採用 |
| `ImprovedAesthetic` | prediction → `[IAP]score_N` (整数 bin) | lib は不採用 |
| `WaifuAesthetic` | prediction → `[WD]score_N` (整数 bin) | lib は不採用。**reference 側に scale 変換抜け bug あり** (ADR 0002 `:248`) |

### Reference 5: `aesthetic_shadow` 4-tier 閾値の出所チェーン (検証記録 2026-06-05)

ADR 0002 (`:139-140`, `:172`) は 4-tier 閾値 (`very aesthetic >= 0.71` 等) を
**「v2 model card 本文に verbatim 記載 (canonical)」** としていた。2026-06-05 に一次ソースを
再検証した結果、**この記述は誤り**で、閾値の出所は shadowlilac の配布元 model card ではなく
**Animagine-XL 3.1 の model card** であることが確定した。本 ADR 0009 が ADR 0002 の当該記述を
訂正する。

#### 検証したソースと結果

| ソース | URL | 4-tier 閾値 | 備考 |
|---|---|---|---|
| **Animagine-XL 3.1** (cagliostrolab) | https://huggingface.co/cagliostrolab/animagine-xl-3.1 | **あり (一次ソース)** | "Aesthetic Tags" 節に `shadowlilac/aesthetic-shadow-v2` で評価した旨 + score range 表を verbatim 記載 |
| Animagine-XL 3.0 (cagliostrolab) | https://huggingface.co/cagliostrolab/animagine-xl-3.0 | なし | 3.1 で導入された基準 (3.0 には未記載) |
| dataset-tag-editor (toshiaki1729) | https://github.com/toshiaki1729/dataset-tag-editor-standalone/blob/main/userscripts/taggers/aesthetic_shadow.py | あり (同値) | コメント `# tags used in Animagine-XL` で出所を明示。`SCORE_N = {'very aesthetic':0.71, 'aesthetic':0.45, 'displeasing':0.27, 'very displeasing':-inf}` を hq に適用 |
| shadowlilac 公式 v2 (オリジナル) | https://huggingface.co/shadowlilac/aesthetic-shadow-v2 | **直接確認不可** | 配布停止 (404)。web.archive.org も取得不可。中身を直接検証できず |
| NeoChen1024 独立ミラー | https://huggingface.co/NeoChen1024/aesthetic-shadow-v2-backup | なし | 別人が作成したミラー。model card に閾値記載なし |
| NEXTAltair ミラー (lib が使用) | https://huggingface.co/NEXTAltair/cache_aestheic-shadow-v2 | あり | ミラー作成者が追記。出所記述なし |

#### Animagine-XL 3.1 model card の verbatim 引用

> "These tags are derived from evaluations made by a specialized ViT (Vision Transformer)
> image classification model, specifically trained on anime data. For this purpose, we
> utilized the model shadowlilac/aesthetic-shadow-v2"

| Aesthetic Tag | Score Range |
|---|---|
| `very aesthetic` | > 0.71 |
| `aesthetic` | > 0.45 & < 0.71 |
| `displeasing` | > 0.27 & < 0.45 |
| `very displeasing` | ≤ 0.27 |

#### 確定した出所チェーン

```
shadowlilac/aesthetic-shadow-v2
    └─ hq / lq の確率 (0–1) を出力するのみ。4-tier 閾値は持たない
         ↓ cagliostrolab が aesthetic-shadow-v2 で自データセットを評価
Animagine-XL 3.1 model card  ★閾値の一次ソース (0.71 / 0.45 / 0.27)
    └─ very aesthetic / aesthetic / displeasing / very displeasing を定義
         ↓ toshiaki1729 が Animagine-XL 基準を採用
dataset-tag-editor (実装リファレンス, ADR 0002)
    └─ コメント "# tags used in Animagine-XL"、hq スコアに SCORE_N 適用
         ↓ image-annotator-lib が実装リファレンスを踏襲
pipeline_scorers.py:_generate_score_labels
    └─ 同じ閾値 0.71 / 0.45 / 0.27 を hq に適用
```

#### 結論

- `very aesthetic` 等の **4-tier 閾値の canonical source は Animagine-XL 3.1 model card**。
  shadowlilac の score model 配布元は閾値を持たない (hq/lq 確率のみ)。
- lib の閾値は「配布元 model card 由来」ではなく「Animagine-XL 基準を toshiaki1729 経由で
  踏襲したもの」。ADR 0002 の "v2 model card 由来 (canonical)" は本 ADR で訂正する。
- **留保**: shadowlilac オリジナル v2 model card は配布停止 + web.archive 取得不可のため
  中身を直接確認できていない。「オリジナルに閾値が無かった」と断定はできないが、独立ミラー
  (NeoChen1024) に閾値が無く、toshiaki1729 が出所を Animagine-XL と明記している状況から、
  閾値の実質的な canonical source は Animagine-XL 3.1 と判断する。

### Reference 6: 実測値 (2026-06-05 probe 実行)

`/tmp/probe_scorer_raw3.out` にて 3 枚の webp 画像 (`file01.webp`, `file02.webp`, `file03.webp`)
に対し 4 scorer を全て実行して得た実測値。CPU モード (CUDA 不可環境)、LoRAIro
`config/annotator_config.toml` を使用。

#### 実測スコア

| Model | 画像 1 (phash b8b036) | 画像 2 (phash cdb542) | 画像 3 (phash e609e8) |
|---|---|---|---|
| `aesthetic_shadow_v1` scores | `hq=0.0877, lq=0.9123` | `hq=0.9078, lq=0.0922` | `hq=0.3815, lq=0.6185` |
| `aesthetic_shadow_v1` score_labels | `['very displeasing']` | `['very aesthetic']` | `['displeasing']` |
| `cafe_aesthetic` scores | `aesthetic=0.9672, not_aesthetic=0.0328` | `aesthetic=0.5251, not_aesthetic=0.4749` | `aesthetic=0.9625, not_aesthetic=0.0375` |
| `cafe_aesthetic` score_labels | `['aesthetic']` | `['aesthetic']` | `['aesthetic']` |
| `ImprovedAesthetic` scores | `aesthetic=6.0978` | `aesthetic=5.8800` | `aesthetic=4.8780` |
| `WaifuAesthetic` scores | `aesthetic=0.9906` | `aesthetic=0.3331` | `aesthetic=0.9999` |

#### 観測値域サマリ

| key | n | min | max |
|---|---|---|---|
| `ImprovedAesthetic.aesthetic` | 3 | 4.8780 | 6.0978 |
| `WaifuAesthetic.aesthetic` | 3 | 0.3331 | 0.9999 |
| `aesthetic_shadow_v1.hq` | 3 | 0.0877 | 0.9078 |
| `aesthetic_shadow_v1.lq` | 3 | 0.0922 | 0.9123 |
| `cafe_aesthetic.aesthetic` | 3 | 0.5251 | 0.9672 |
| `cafe_aesthetic.not_aesthetic` | 3 | 0.0328 | 0.4749 |

> 初回 probe (PR #140/#141 適用前) では WaifuAesthetic が全画像 `0.5565` の同値だった。
> Issue #142 (PR #143) で `fc{n}.weight` キー検出とキーリネーム対応を修正済み。
> 上記は修正後の有効値。

### Reference 7: pipeline 系スコアラー変換の変遷 (cafe_aesthetic の中途半端な状態)

ADR 0002 `:55-66` で実装リファレンスとした `toshiaki1729/dataset-tag-editor` は
cafe 生値 (`0-1` 確率) を整数 bin に変換していた (`[CAFE]score_N`)。lib は当初この変換を
採用せず、後で一部を採用した結果、現在は「toshiaki の整数 bin でも生値でもない中間状態」
になっている。

#### 変遷チェーン

```
cafeai/cafe_aesthetic (HF Pipeline)
    └─ softmax → 'aesthetic'/'not_aesthetic' 確率 0-1 を返す (生値)
         ↓ toshiaki1729 実装 (ADR 0002 の実装リファレンス)
dataset-tag-editor: score × 10 → floor → [CAFE]score_N (整数 bin)
    └─ not_aesthetic = 1.0 - aesthetic で再構成 (lib が同様の再構成を採用した背景)
         ↓ image-annotator-lib 現在実装 (pipeline_scorers.py:101-153)
pipeline_scorers.py: aesthetic 確率をそのまま scores に入れ、
    not_aesthetic = 1.0 - aesthetic で再構成 (toshiaki 再構成だけ踏襲、整数 bin 化は不採用)
    argmax で 'aesthetic' / 'not_aesthetic' を score_labels に設定
```

#### 現在の状態評価

`cafe_aesthetic` の現実装は:
- `not_aesthetic = 1.0 - aesthetic` は toshiaki 由来の再構成であり、pipeline が
  `softmax(aesthetic, not_aesthetic)` を返す場合は同値だが、**pipeline が 2 class を
  直接返す場合は冗長または誤差を含む可能性がある**
- `score_labels` は argmax (`>0.5` で `aesthetic`) — これも toshiaki 非採用、生値返しへの
  中途半端な残留物

**生値返しを徹底するなら**、`not_aesthetic` を drop して `aesthetic` 確率のみ返し、
`score_labels` も廃止か `capabilities` から `score_labels` を外すべき。
ただし LoRAIro#626 の正規化方針が決まった段階で一括整理する
(本 ADR では現状の事実記録に留める)。

#### CLIP 系との比較 (中途半端マップ)

| 系統 | モデル | `scores` | `score_labels` | 設計方針適合度 |
|---|---|---|---|---|
| CLIP 系 | `ImprovedAesthetic` | 生値、無変換 | なし | ✅ 生値のみ方針に完全準拠 |
| CLIP 系 | `WaifuAesthetic` | Sigmoid 変換済み (config 由来) | なし | ✅ 生値のみ方針に準拠 (head 問題は別) |
| Pipeline 系 | `aesthetic_shadow_v1/v2` | 生確率、無変換 | Animagine-XL 由来閾値で 4-tier | △ `score_labels` に配布元 canonical でない閾値 |
| Pipeline 系 | `cafe_aesthetic` | 生確率 + 再構成 `not_aesthetic` | argmax ラベル | △ `not_aesthetic` 再構成は toshiaki 踏襲で根拠不明確 |

### Reference 8: `score_scales` メタデータ実装 (Issue #144, 本 ADR 確定の根拠)

本 reference (Reference 2/3 の値域) を lib が機械可読メタデータとして宣言する実装を
Issue #144 で追加した。**正規化は consumer 責務とし、lib は値域メタデータの提供のみを行う**
(ADR 0002 が予約した「lib 側で正規化 scores を提供するか」の Open Question に対し、
「提供しない。値域 reference だけ宣言する」と回答)。

#### 追加した型 (`core/types.py`)

```python
@dataclass(frozen=True)
class ScoreScale:
    range: tuple[float, float]      # 生値の理論値域 (min, max)
    higher_is_better: bool = True

# UnifiedAnnotationResult に追加 (default 付き = 非破壊追加)
score_scales: dict[str, ScoreScale] | None = None
```

#### 各 scorer の `score_scales` 宣言

| Model | `score_scales` | 宣言場所 |
|---|---|---|
| `aesthetic_shadow_v1/v2` | `{"hq": (0–1, ↑), "lq": (0–1, ↓)}` | `pipeline_scorers.py:AestheticShadow.SCORE_SCALE` |
| `cafe_aesthetic` | `{"aesthetic": (0–1, ↑), "not_aesthetic": (0–1, ↓)}` | `pipeline_scorers.py:CafePredictor.SCORE_SCALE` |
| `ImprovedAesthetic` | `{"aesthetic": (1–10, ↑)}` | `scorer_clip.py:ImprovedAesthetic.SCORE_SCALE` |
| `WaifuAesthetic` | `{"aesthetic": (0–1, ↑)}` | `scorer_clip.py:WaifuAesthetic.SCORE_SCALE` |

(↑ = `higher_is_better=True`, ↓ = `False`)

実装方針: 各 scorer subclass が **クラス属性** `SCORE_SCALE: dict[str, ScoreScale]` を宣言し、
`_format_predictions` (pipeline 系) / `ClipBaseAnnotator._format_predictions` (CLIP 系) が
結果構築時に `score_scales` へ載せる。CLIP 系は `scores` を返す場合のみ `score_scales` も
添える (両 field を整合させる)。

#### consumer (LoRAIro) への含意

- CLIP 系 2 モデルが同じ key `aesthetic` を使い値域が違う問題 (Improved 非有界 1–10 /
  Waifu 0–1) は、`score_scales["aesthetic"].range` で機械的に判別できるようになった。
- 共通スケールへの写像方式は依然として consumer 側で決める (LoRAIro#626)。lib は
  「この生値はこの値域で、大きいほど良い」という事実のみを構造化して渡す。

## Rationale

- **正規化を急いで決めない**: 値域・有界性・配布元保証の有無が model ごとに異なり (確率 / 有界
  regression / 非有界 regression の 3 群)、reference を曖昧にしたまま方式を選ぶと ADR 0022
  (LoRAIro) のような「表と本文で値域が食い違う」状態を再生産する。まず事実を固める。
- **lib SSoT として一箇所に集約**: 値域は model card 単独では確認できず `config.json` /
  訓練データ仕様 / 推論コードを併読して初めて確定する (ADR 0002 `:148-163`)。次に scorer を
  追加 / 疑う人が再調査せずに済むよう、確定済み reference を ADR に残す。

## Consequences

### 良い点
- 正規化方式 (LoRAIro#626) の議論が、確定した値域 reference の上で行える。
- scorer 追加時に「生の戻り → lib 変換」の記録テンプレートとして使える。
- 実測値 (Reference 6) が値域の「実際の使用範囲」を裏付け、理論値域との乖離を把握できる。
  (例: `ImprovedAesthetic` の理論値域は 1–10 だが 3 画像では 4.88–6.10 に収まった)

### 悪い点・制約
- Status `Proposed` のまま放置すると reference が陳腐化する。scorer の活性化 / head 構造を
  変更したら本 ADR を更新する運用が必要。
- CLIP 系 2 モデルが同じ `scores` key (`aesthetic`) を使い値域が違うため、consumer が
  model_name を見ずに値域を仮定すると誤る (本 reference で明示)。
- **WaifuAesthetic は実測値が無効** (Reference 6): head 構造推測失敗で 3 枚全て同値 `0.5565`。
  修正前は WaifuAesthetic の `scores` を信用しない。
- **pipeline 系の中途半端状態** (Reference 7): `cafe_aesthetic` の `not_aesthetic` 再構成と
  `aesthetic_shadow` の Animagine-XL 由来閾値は、生値方針に完全準拠していない。
  LoRAIro#626 の正規化方針確定後に整理が必要。

## Open Questions (正規化 contract で決めること)

本 reference を踏まえ、後続で以下を決定する:

1. **共通スケールの基準**: 0–1 / 0–10 / 0–100 のどれに揃えるか。LoRAIro の手動編集スコア
   (0.0–10.0) / スライダー (内部 0–1000) との整合をどう取るか (LoRAIro#626)。
2. **群ごとの写像**:
   - 有界確率 (shadow `hq` / cafe `aesthetic`): どの key を代表値にするか、線形に伸ばすか。
   - 有界 regression (Waifu 0–1): 確率群と同じ写像でよいか。
   - **非有界 regression (Improved ~1–10, clamp なし)**: clamp [1,10] / min-max / sigmoid 等の
     有界化が必須。境界値の根拠をどこに置くか。
3. ~~**正規化の責務境界**~~ (Issue #144 で決定済み): lib は正規化 `scores` を提供せず、
   値域メタデータ `score_scales` のみを宣言する (Reference 8)。共通スケールへの写像は
   consumer (LoRAIro) 側の責務。生値 `scores` と値域 `score_scales` の両方を返す。
4. **データ依存の可否**: percentile / z-score のようなデータセット相対正規化を許容するか
   (再現性・単一画像推論との trade-off)。LoRAIro#626 の検討表参照。
5. **将来 scorer**: ImageReward (正規分布 μ=0.167, σ=1.033 / 非有界) 等を導入する場合の
   拡張余地 (LoRAIro ADR 0022 将来候補)。
6. ~~**WaifuAesthetic head 修正**~~: Issue #142 / PR #143 で解決済み。Reference 6 の実測値を有効値に更新。
7. **pipeline 系整理** (Reference 7): 生値方針に完全準拠させるか、現行の
   `score_labels` / `not_aesthetic` 再構成を明示的に仕様化するか決定する。

## Related

- **ADR 0002** (Score Model Output Contract) — 正規化を本 ADR に予約した元 contract。
  本 ADR が `:172` の「4-tier 閾値 = v2 model card 由来 (canonical)」記述を訂正 (Reference 5)
- **LoRAIro#626** — スコア表示の尺度合わせ方針 (consumer 側要件)
- **LoRAIro ADR 0022** — Aesthetic Score Predictor Survey (値域の散文記述、本 ADR で実装裏取り)
- **Issue** NEXTAltair/image-annotator-lib#66 — scorer output contract の発端

### Source URLs (2026-06-05 検証)

- 閾値一次ソース (Animagine-XL 3.1): https://huggingface.co/cagliostrolab/animagine-xl-3.1
- Animagine-XL 3.0 (閾値なし、参考): https://huggingface.co/cagliostrolab/animagine-xl-3.0
- 実装リファレンス (toshiaki1729): https://github.com/toshiaki1729/dataset-tag-editor-standalone/blob/main/userscripts/taggers/aesthetic_shadow.py
- shadowlilac v1: https://huggingface.co/shadowlilac/aesthetic-shadow
- shadowlilac v2 オリジナル (配布停止): https://huggingface.co/shadowlilac/aesthetic-shadow-v2
- NeoChen1024 独立ミラー (閾値なし): https://huggingface.co/NeoChen1024/aesthetic-shadow-v2-backup
- NEXTAltair ミラー (lib 使用): https://huggingface.co/NEXTAltair/cache_aestheic-shadow-v2
- ImprovedAesthetic (christophschuhmann): https://github.com/christophschuhmann/improved-aesthetic-predictor
- WaifuAesthetic (waifu-diffusion): https://github.com/waifu-diffusion/aesthetic
- WaifuAesthetic 重み (hakurei): https://huggingface.co/hakurei/waifu-diffusion-v1-4/tree/main/models
- cafe_aesthetic: https://huggingface.co/cafeai/cafe_aesthetic
