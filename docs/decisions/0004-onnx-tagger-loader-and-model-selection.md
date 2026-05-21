# ADR 0004: ONNX Tagger Loader and Model Selection

- **日付**: 2026-05-21
- **ステータス**: Accepted

## Context

ADR 0003 で `ratings` field と model-native label contract を定義し、最初の rating-only
adapter として `deepghs/anime_rating` を追加した。次の候補として、rating だけでなく
general / character tags も返せる Camie 系モデルを検討した。

既存の WDTagger 系 adapter は、Hugging Face repo から `model.onnx` と `selected_tags.csv`
を取得し、`ONNXBaseAnnotator` / `ONNXLoader` 経由で推論する。Camie 系モデルも ONNX だが、
metadata は CSV ではなく JSON であり、ファイル名も `model_initial.onnx` /
`model_initial_metadata.json` のように異なる。そのため、既存の `download_onnx_tagger_model()`
は実質 WDTagger 専用になっている。

また、モデル選択では以下の制約がある。

- GUI のモデル選択肢を増やしすぎると使い勝手が悪くなる。
- rating では精度を重視する。
- モデル重みは同梱しない。
- GPL-3.0 モデルを扱う場合でも、GPL 実装コードのコピーやモデル同梱は避ける。
- LoRAIro canonical rating への mapping は consumer 側で行う。

## Decision

1. ONNX tagger の取得処理を WDTagger 専用から汎用化する。
2. `ONNXLoader` / `ModelLoad.load_onnx_components()` は、ONNX model filename と metadata
   filename を optional parameter として受け取れるようにする。
3. `ONNXBaseAnnotator` は class variable または hook で model/metadata filename を宣言する。
4. WDTagger 系は既存互換の default として `model.onnx` / `selected_tags.csv` を使う。
5. Camie 系は WDTagger と同じ「モデル同梱なし、Hugging Face から ONNX と metadata を取得し、
   adapter が metadata から label/category を復元する」方式で採用する。
6. Camie の初期対応は `rating`, `general`, `character` のみに限定する。
   `artist`, `copyright`, `meta`, `year` は初期出力に含めない。
7. Camie 公式推論コードや `dghs-imgutils` の実装はコピーしない。公開仕様と metadata 形式に基づき、
   image-annotator-lib 側で adapter を独自実装する。
8. Camie model license が GPL-3.0 であることを docs / model specs に明記する。
9. Camie 以外の追加 rating model として `deepghs/anime_rating` の `caformer_s36_plus`
   variant を採用する。これは既存 `AnimeRatingAnnotator` と同じ ONNX 実装で扱う。

## Model Selection

標準 rating-only path は `deepghs/anime_rating` 系を維持する。

- `mobilenetv3_sce_dist`: 高速・大量処理用。
- `caformer_s36_plus`: 精度優先用。`anime_rating` の公開指標上、最も高い accuracy を持つため
  採用する。

Camie 系は rating-only model ではなく、WDTagger 後継候補に近い large ONNX tagger として扱う。
rating も返せるが、主な採用理由は `rating`, `general`, `character` をまとめて高品質に返せる点である。

その他の追加候補は現時点では採用しない。

- `deepghs/eattach_sankaku_rating`: `anime_rating` と同じ `sankaku3` で用途が重なり、公開指標上の
  優位性も弱い。
- `ggg4mless/RateBooru_Efficient`: MIT だが TensorFlow/Keras 依存が増え、既存候補との差別化が弱い。
- binary NSFW classifier: rating 保存モデルではなく、routing / filtering signal として別 issue で扱う。

詳細な model card links、metrics、license は
`docs/model-specs/rating-model-candidates.md` に記録する。

## Rationale

### なぜ loader を汎用化するか

`download_onnx_tagger_model()` は `model.onnx` と CSV metadata を前提にしており、WDTagger 以外の
ONNX tagger を追加するたびに専用 loader を増やすと重複が増える。model filename と metadata
filename を adapter 側で宣言できるようにすれば、WDTagger 互換を保ったまま JSON metadata 型の
tagger も同じロード経路に乗せられる。

### なぜ Camie を WDTagger と同じ実装形態で採用するか

Camie は GPL-3.0 model だが、モデル重みや metadata を package に同梱せず、GPL 実装コードも
コピーしない。WDTagger と同様に外部 repo の ONNX model を利用者環境で取得し、image-annotator-lib
側の独自 adapter が推論結果を解釈する形にする。この実装形態であれば、モデル自体の license は
明示しつつ、library 側のコードを GPL 由来コードに強く依存させない。

### なぜ Camie を rating-only ではなく large tagger として扱うか

Camie は約 70,000 tag の multi-label classifier であり、rating だけを返す軽量モデルではない。
rating 精度だけを目的に標準モデルを増やすより、WDTagger 系の代替・補完として `rating` /
`general` / `character` を返す大型 tagger と位置づける方が利用者にとって分かりやすい。

### なぜ出力カテゴリを絞るか

Camie docs では `rating`, `general`, `character` は有用だが、`year`, `meta`, `artist`,
`copyright` は精度が限定的とされている。初期対応で低精度カテゴリまで返すと、利用者が tag 品質を
誤解しやすくなるため、初期出力は高品質なカテゴリに絞る。

### なぜ `anime_rating_caformer_s36_plus` を採用するか

`anime_rating_mobilenetv3_sce_dist` は高速・軽量で大量処理に向くが、rating では精度を重視する。
同じ `deepghs/anime_rating` repo 内で公開されている `caformer_s36_plus` は、model card 上の
accuracy が最も高い。license も同じ MIT で、出力 scheme も `sankaku3` のままなので、既存
`AnimeRatingAnnotator` に variant 設定を追加するだけで扱える。GUI では高速枠と精度優先枠を
用途で分けられる。

## Consequences

### 良い点

- ONNX tagger 追加時の loader 重複が減る。
- WDTagger の既存挙動を保ったまま Camie 系に対応できる。
- GUI の標準選択肢を rating-only と large tagger の用途別に整理できる。
- Camie の GPL-3.0 model license を明示しつつ、モデル同梱や GPL コードコピーを避けられる。

### 悪い点・トレードオフ

- `ModelLoad.load_onnx_components()` / `ONNXLoader` の引数が増える。
- `ONNXComponents` は `csv_path` 前提から `metadata_path` 前提へ移行する必要がある。
- Camie は重いため、runtime validation や GUI default には向かない。
- GPL-3.0 model を扱う以上、利用者向け docs で license 注意書きが必要になる。

### 実装ルール

- 既存互換のため、移行期間は `ONNXComponents` に `metadata_path` と `csv_path` の両方を持たせてよい。
- 新規 ONNX tagger は `metadata_path` を参照する。
- Camie adapter は公式推論コードをコピーせず、独自実装する。
- Camie adapter は `TaskCapability.TAGS` と `TaskCapability.RATINGS` を宣言し、rating label を
  `tags` に混ぜない。

## Related

- ADR 0003: Rating Model Output Contract
- Model specs: `docs/model-specs/rating-model-candidates.md`
