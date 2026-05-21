# ADR 0003: Rating Model Output Contract

- **日付**: 2026-05-21
- **ステータス**: Accepted

## Context

WaifuDiffusion Tagger 系モデルは、content tag だけでなく `general` / `sensitive` /
`questionable` / `explicit` のような rating category も出力する。さらに、rating / NSFW 判定に
使える新規候補モデルとして以下がある。

- `deepghs/anime_rating`
- `deepghs/eattach_sankaku_rating`
- `Camais03/camie-tagger-v2`
- `ggg4mless/RateBooru_Efficient`
- binary NSFW / explicit-content classifier

これらの label scheme はモデルごとに異なる。LoRAIro は `PG` / `PG-13` / `R` / `X` / `XXX`
という canonical rating を持つが、それは consumer 固有の policy であり、image-annotator-lib の
共通 contract ではない。

ADR 0002 は score-derived categorical label を `score_labels` として定義し、model-specific key
や scale を無理に正規化しない方針を採った。rating model output も同じく、まず配布元・モデル
固有の label を保持する必要がある。

## Decision

1. 新規 capability `TaskCapability.RATINGS` を定義する。
2. `UnifiedAnnotationResult` に `ratings: list[RatingPrediction] | None` を追加する。
3. `ratings` は model-native な top prediction を返す。LoRAIro canonical rating へ変換しない。
4. full distribution が必要な場合は `raw_output` に保持する。`ratings` は downstream が扱いやすい
   summary field とする。
5. rating 対応を実装した model adapter だけが `TaskCapability.RATINGS` を宣言する。rating 能力の
   ないモデルに `ratings` は期待しない。

### RatingPrediction

`RatingPrediction` は以下の情報を持つ。

| field | 型 | 意味 |
|---|---|---|
| `raw_label` | `str` | モデルが返した rating label。例: `general`, `questionable`, `r15`, `NSFW` |
| `confidence_score` | `float | None` | `raw_label` の confidence。取得できないモデルでは `None` |
| `source_scheme` | `str` | label scheme id。例: `danbooru4`, `e6213`, `sankaku3`, `binary_nsfw` |

`normalized_rating` / `canonical_rating` / `lorairo_rating` のような consumer policy field は持たせない。

### Source schemes

初期対応の scheme id は以下とする。

| `source_scheme` | labels | 想定モデル |
|---|---|---|
| `danbooru4` | `general`, `sensitive`, `questionable`, `explicit` | WaifuDiffusion Tagger / `Camais03/camie-tagger-v2` |
| `danbooru3` | `general`, `questionable`, `explicit` | `ggg4mless/RateBooru_Efficient` 等 |
| `e6213` | `safe`, `questionable`, `explicit` | Z3D E621Tagger 系 |
| `sankaku3` | `safe`, `r15`, `r18` | `deepghs/anime_rating`, `deepghs/eattach_sankaku_rating` |
| `binary_nsfw` | `SFW`, `NSFW` | generic NSFW classifier |
| `explicit_content` | model-specific explicit-content labels | generic explicit-content classifier |

binary NSFW 候補モデルの model card:

- `AdamCodd/vit-base-nsfw-detector`: https://huggingface.co/AdamCodd/vit-base-nsfw-detector
  - 二値: `sfw` / `nsfw`
  - 実写・3D・drawings を含む約 25,000 images で fine-tune。model card では generated image では
    性能が下がると明記されている。
- `AdamCodd/vit-nsfw-stable-diffusion`: https://huggingface.co/AdamCodd/vit-nsfw-stable-diffusion
  - 二値: `sfw` / `nsfw`
  - Stable Diffusion 生成画像向け。access gate と非商用系 license 条件があるため採用前確認が必要。
- `Falconsai/nsfw_image_detection`: https://huggingface.co/Falconsai/nsfw_image_detection
  - 二値: `normal` / `nsfw`
  - NSFW image classification 用の ViT fine-tune。
- `CaveduckAI/nsfw-classifier`: https://huggingface.co/CaveduckAI/nsfw-classifier
  - 二値: `SFW` / `NSFW`
  - ConvNeXt 系。model card に validation accuracy と label id が明記されている。

これらは rating を細分化するモデルではなく、`binary_nsfw` scheme の routing / filtering signal として
扱う。Civitai 5段階などへの変換は consumer 側の責務であり、library では行わない。

### Existing WDTagger behavior

WDTagger / Z3D E621Tagger など、既に rating category を内部出力しているモデルは、
content tags とは別に `ratings` を返す。rating category を `tags` に混ぜない。

既存実装で full category distribution を `raw_output` に保持している場合、それは維持する。
`ratings` には top-1 を入れる。

top-1 は confidence が最大の label とする。confidence が取得できないモデルでは adapter が最も
妥当な 1 件を返し、`confidence_score=None` とする。推定値を `1.0` や `0.0` で埋めない。

`raw_label` は `source_scheme` 内の公式表記に寄せる。例えば provider や model が `Safe` /
`SAFE` / `safe` のように表記揺れする場合、adapter 内で `safe` に揃える。ただし LoRAIro
canonical rating には変換しない。

### WebAPI models

WebAPI 経由で rating を依頼する場合も、library は prompt / provider の model-native label を
`raw_label` として返す。LoRAIro canonical rating への mapping は行わない。

ただし WebAPI rating は初期実装の範囲外とし、別 issue で扱う。

## Rationale

### なぜ LoRAIro rating に変換しないか

`PG` / `PG-13` / `R` / `X` / `XXX` は LoRAIro の保存・検索・除外判定用の policy であり、
汎用 library が固定すべき意味ではない。同じ `r15` を `PG-13` と見るか `R` と見るかは
consumer の運用方針で変わる。

### なぜ `tags` に入れないか

`tags` は画像内容を表す content tag 用である。rating label を `tags` に混ぜると、下流が
`1girl` / `outdoor` と `questionable` / `explicit` を同じ種類の検索・export 対象として扱って
しまう。rating は専用 field と capability で分離する。

### なぜ `ratings` field を追加するか

`ratings` を専用 field として追加することで、モデルが rating を返すことを明示できる。
既存の `tags` や `score_labels` に混ぜるより意味が分かれ、ライブラリ利用側も
`result.ratings` だけを見れば rating 保存・routing 処理へ渡せる。

`raw_output` だけに rating を置く案では、利用側が model id ごとに
`raw_output["category_scores"]["ratings"]` や `raw_output["labels"]` のような
model-specific な構造を解析する必要がある。専用 field を持たせれば、その解析責務を
各 model adapter 内に閉じ込められる。

また、`TaskCapability.RATINGS` と `ratings` field を対応させることで、`ratings` を返すモデルは
capability に `RATINGS` を宣言しなければならない、という既存の出力検証パターンに乗せられる。
これにより「rating を返しているのに capability に出ていない」「rating capability なのに
`tags` に混ぜている」といった contract 不整合をテストで検出できる。

### なぜ top-1 field と raw_output を分けるか

多くの consumer は保存・filter 用に最終 label だけを必要とする。一方で、threshold 調整や UI
表示では full distribution が必要になる。`ratings` は安定 contract として top-1 を提供し、
詳細は model-specific `raw_output` に保持する。

WDTagger 系のように category probability を持つモデルでは、full distribution は
`raw_output["category_scores"]["ratings"]` に保持する。二値 NSFW classifier など別形式のモデルでも、
`raw_output` に model-native な全出力を残す。

### 却下した選択肢

| 案 | 却下理由 |
|---|---|
| library が `PG` / `R` 等へ変換する | consumer 固有 policy を library に固定してしまう |
| rating label を `tags` に混ぜる | content tag と safety/rating label が区別不能になる |
| binary NSFW 用の専用 field を作る | binary NSFW は rating / routing signal の一種であり、field を増やすと consumer contract が分散する |
| full distribution を `ratings` に全展開する | stable summary field と model-specific raw output の境界が曖昧になる |

## Consequences

### 良い点

- model-native label と confidence を失わない。
- LoRAIro 以外の consumer が自分の policy で mapping できる。
- WDTagger 系の rating category を content tags から分離できる。
- ADR 0002 と同様に、library はモデル出力の意味を保持し、正規化 policy を consumer に任せる。

### 悪い点・トレードオフ

- downstream consumer は `source_scheme` + `raw_label` の mapping を自前で持つ必要がある。
- `source_scheme` の命名を増やすときは、モデル追加 issue / PR で label 一覧を明示する必要がある。
- full distribution は `raw_output` の model-specific 形式を参照するため、横断的な UI には追加設計が必要。

### 運用ルール

- `TaskCapability.RATINGS` を宣言したモデルは `ratings is not None` を返す。
- rating 能力がないモデルは `TaskCapability.RATINGS` を宣言しない。利用側も `ratings` を期待しない。
- rating 抽出実装が入っていないモデルに capability だけを先に足さない。
- `ratings` を返すモデルは、`tags` に rating category を混ぜない。
- `raw_label` は配布元の label spelling を原則保持する。ただし provider が空白・大小文字を揺らす場合は
  model adapter 内で scheme ごとに定義した spelling に寄せる。
- `source_scheme` は model id ではなく label scheme id とする。同じ scheme を複数モデルで共有してよい。
- LoRAIro canonical rating への mapping は image-annotator-lib では実装しない。

## Related

- ADR 0002: Score Model Output Contract
- NEXTAltair/image-annotator-lib#79
- NEXTAltair/image-annotator-lib#80
- NEXTAltair/image-annotator-lib#81
- NEXTAltair/image-annotator-lib#82
- NEXTAltair/LoRAIro#333
- LoRAIro ADR 0031: Model-Native Rating Mapping and Storage
