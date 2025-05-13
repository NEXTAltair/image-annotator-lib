# PydanticAI導入検討・設計案

## 1. 検討背景・目的
- 画像アノテーションWebAPI系アノテーターの型安全性・保守性・拡張性向上
- 依存性注入・構造化出力・エージェント設計のモダン化
- Logfire等によるトレーシング・監視の容易化
- 既存API・クラス構造・テストとの互換性維持

## 2. 現状の課題
- 設定値・APIクライアントの管理が分散しがち
- APIレスポンスの型安全なバリデーションが弱い
- サブクラスごとの重複・責務分離が不十分
- テスト容易性・拡張性に課題

## 3. PydanticAI導入の設計方針
- 既存の`BaseAnnotator`/`WebApiBaseAnnotator`/サブクラスのAPI・インターフェースは維持
- 内部実装のみPydanticAI化(依存性・レスポンス管理、型バリデーション、エージェント設計)
- 依存性(APIクライアント・設定値)をPydanticモデルで一元管理
- APIレスポンスをPydanticモデルでバリデーションし、既存TypedDict型に変換して返却
- 例外・エラー処理は既存カスタム例外を維持しつつ、必要に応じPydanticAIのエラーラッピングを追加
- Logfire等のInstrumentationは開発・運用時に任意で有効化

## 4. 主要変更点・設計例
### 4.1 依存性注入のPydanticモデル化
```python
from pydantic import BaseModel
class OpenAIDependencies(BaseModel):
    api_key: str
    model_id: str
    timeout: int = 60
```
### 4.2 レスポンスの型定義・RawOutputとWebApiFormattedOutputの利用 {#response-types}
```python
from typing import TypedDict, Any
from pydantic import BaseModel

# Pydanticモデル (内部バリデーション用)
class AnnotationSchema(BaseModel):
    tags: list[str] | None = None
    captions: list[str] | None = None
    score: float | None = None

# _run_inference の戻り値型 (統一)
class RawOutput(TypedDict, total=False):
    response: AnnotationSchema | None  # バリデーション済みPydanticモデル or None
    error: str | None              # エラーメッセージ(なければNone)

# _format_predictions の戻り値型 (統一)
class WebApiFormattedOutput(TypedDict, total=False):
    annotation: dict[str, Any] | None # .model_dump() された辞書 or None
    error: str | None               # エラーメッセージ(なければNone)

# _run_inference の戻り値例
results: list[RawOutput] = [
    {"response": AnnotationSchema(tags=["cat"], captions=["A cat"], score=0.9), "error": None},
    {"response": None, "error": "API Error: Connection timeout"},
]

# _format_predictions の戻り値例
formatted_results: list[WebApiFormattedOutput] = [
    {"annotation": {"tags": ["cat"], "captions": ["A cat"], "score": 0.9}, "error": None},
    {"annotation": None, "error": "API Error: Connection timeout"},
]

```
- **`_run_inference` の戻り値:** すべてのWeb API系アノテーターで **`list[RawOutput]`** に統一。(ローカルモデルアノテーターは現時点では異なる可能性があるため要確認 TODO:)
    - `RawOutput` は `src/image_annotator_lib/core/types.py` で定義された `TypedDict` (`total=False`)。
    - `response` キーには、バリデーション済みの `AnnotationSchema` Pydanticオブジェクト、またはAPIエラー等でレスポンスが得られなかった場合は `None` が格納される。
    - `error` キーには、処理中に発生したエラーメッセージ(APIエラー、パースエラー等)、または正常時は `None` が格納される。2025-05-13の改修により、このエラーメッセージには原則として発生した例外のクラス名が含まれるようになった。
- **`_format_predictions` の戻り値:** `WebApiBaseAnnotator` で共通実装され、すべてのWeb API系アノテーターで **`list[WebApiFormattedOutput]`** に統一。
    - `WebApiFormattedOutput` は `src/image_annotator_lib/core/types.py` で定義された `TypedDict` (`total=False`)。
    - `annotation` キーには、`RawOutput` の `response` (`AnnotationSchema`) を `.model_dump()` で変換した **辞書**、またはエラー時は `None` が格納される。辞書形式なのは、API以外のモデルとの互換性を保つため。
    - `error` キーには、`RawOutput` から引き継いだエラーメッセージ、または正常時は `None` が格納される。2025-05-13の改修により、このエラーメッセージには原則として発生した例外のクラス名が含まれるようになった。
- **Pydanticモデル (`AnnotationSchema` 等) の役割:** 主に `_run_inference` 内でのAPIレスポンスのバリデーションと型安全なデータ保持に使用する。外部インターフェース (`_format_predictions` の戻り値) では、互換性のために辞書型に変換する。

### 4.3 Annotatorクラスの内部実装例
```python
class OpenAIAnnotator(WebApiBaseAnnotator):
    def __init__(self, deps: OpenAIDependencies):
        self.deps = deps
    def _run_inference(self, processed_images: list[str]) -> dict:
        ...
    def _format_predictions(self, raw_response: dict) -> OpenAIAnnotationOutput:
        return OpenAIAnnotationOutput(**raw_response)
    def _generate_tags(self, formatted_output: OpenAIAnnotationOutput) -> list[str]:
        return formatted_output.tags
```
### 4.4 既存APIとのブリッジ
- predict()や_generate_result()の戻り値は既存TypedDict型を維持
- 内部でPydanticモデル→TypedDictへの変換を行う

## 5. トレードオフ・注意点
- 利点: 型安全・保守性・テスト容易性向上、重複削減、エラー処理一元化
- 注意点: モデル設計の煩雑化、TypedDict型との相互変換コスト、PydanticAIのAPI追従

## 6. 段階的導入案(BDDテスト互換性担保を最優先)

### ステップ1: 既存BDDテストの現状把握・動作確認

- **目的:** 既存のWebApiBaseAnnotatorおよび各サブクラス(OpenAI/Anthropic/Google等)が、現状のBDDテストで仕様通りに動作しているかを正確に把握する。

- **手順:**
  1. **テスト資産の洗い出し**  
     `tests/features/`配下の全featureファイルとstep定義をリストアップし、どのAPI・モデル・ケース(正常/異常)がカバーされているか確認する。
  2. **テスト実行準備**  
     必要なAPIキー・設定ファイル・サンプル画像など、テスト実行に必要な環境が揃っているか確認する。
  3. **BDDテストの一括実行**  
     `pytest`等で全BDDテストを実行し、結果(パス/失敗/スキップ)を記録する。
  4. **失敗時の詳細分析**  
     失敗したテストがあれば、エラー内容・再現手順・失敗原因(実装バグ/仕様逸脱/テスト不備等)を明確にし、現状の仕様との差分を整理する。
  5. **現状仕様の明文化**  
     テスト結果・分析内容をもとに、現状の「仕様・制約・既知の問題点」を設計ドキュメント等に記録する。

- **観点:**
  - すべての主要API・モデル・異常系がテストでカバーされているか
  - テストの再現性・安定性(外部APIの仕様変更やネットワーク障害等の影響も考慮)
  - 失敗時の原因分析・記録が十分か

- **記録例:**
  - 2025-05-11: 全BDDテストがパス。OpenAI Visionのoutput_textパース方式修正により、APIレスポンス形式の揺れにも対応できることを確認。
  - 2025-05-12: Google Gemini APIの空レスポンス時にテストがスキップされることを確認。仕様として許容する方針をドキュメント化。
  - 2025-05-13:
    - APIキー未設定時のBDDテスト(`test_apiキーが未設定の場合は認証エラーが発生する`)が当初`AssertionError`で失敗。原因は、`api.py`の`_handle_error`が生成するエラーメッセージに例外の型名(`ApiAuthenticationError`)が含まれていなかったため。`_handle_error`を修正し、エラーメッセージに例外型名を含めることでテストパスを確認。
    - APIタイムアウト時およびAPIエラーレスポンス時のBDDテスト(`test_apiリクエストがタイムアウトした場合は適切なエラーを返す`, `test_apiからエラーレスポンスが返された場合は適切に処理する`)の安定性を向上。`webapi_annotate_steps.py`の関連するGivenステップを修正し、API呼び出しをモックして意図的なエラーを送出するように変更。また、Thenステップのアサーションを調整し、大文字・小文字を区別せずにエラーメッセージを検証するように修正。これにより、全BDDテストがパスすることを確認。

### ステップ2: 依存性・レスポンスのPydanticモデル定義
- **目的:** 既存の設定値・APIクライアント・レスポンス型をPydanticモデル/データクラスで明示的に定義し、型安全な設計に移行する準備。
- **手順:**
  1. まずOpenAI系(annotator_webapi/openai_api.py等)から着手。
  2. 依存性(APIキー、モデルID、タイムアウト等)をPydanticモデルまたはdataclassで定義。
  3. レスポンス(タグ、キャプション、スコア等)もPydanticモデルで定義。
  4. 既存のTypedDict型との相互変換関数も用意。

### ステップ3: WebApiBaseAnnotatorの内部実装をPydanticAI化
- **目的:** 依存性注入・レスポンスバリデーション・ツール設計をPydanticAIベースにリファクタ。
- **手順:**
  1. 既存のWebApiBaseAnnotatorの内部で、依存性・レスポンスをPydanticモデルで管理。
  2. `_format_predictions` メソッドを `WebApiBaseAnnotator` に実装し、`list[RawOutput]` を `list[WebApiFormattedOutput]` に変換する共通ロジックを集約。(2025-08-07頃 実施済み)
     - この際、`RawOutput` の `response` (Pydanticモデル `AnnotationSchema`) を `.model_dump()` して `WebApiFormattedOutput` の `annotation` (辞書) に格納するようにした。
  3. 必要に応じてAgent/tool/system_prompt等のPydanticAI APIを導入。
  4. 外部インターフェース(predict()等)は既存のまま維持し、テスト互換性を保つ。

### ステップ4: サブクラス(OpenAI/Anthropic/Google等)の順次移行
- **目的:** 各APIごとのサブクラスもPydanticAIベースに統一。
- **手順:**
  1. 各サブクラスから `_format_predictions` メソッドを削除し、基底クラスの実装を利用するように変更。(2025-08-07頃 実施済み)
  2. 各サブクラスの `_run_inference` の戻り値を `list[RawOutput]` (responseが `AnnotationSchema | None`) に統一。(2025-08-07頃 実施済み)

### ステップ5: BDDテストによる互換性・完成度検証
- **目的:** PydanticAI化後も既存BDDテストが全てパスすることを必ず確認し、仕様互換性・完成度を担保。
- **手順:**
  1. 各リファクタ段階ごとにBDDテストを必ず実行。
  2. テストが失敗した場合は、原因を分析し、仕様逸脱がないよう修正。
  3. すべてのテストがパスした時点で「互換性・完成度が担保された」と判断。
- **補足:**
  - pytestをテストハーネスとして利用。
  - PydanticAIのAgent.overrideやTestModel/FunctionModelを活用し、LLM呼び出しをモック化したユニットテストも併用可能。
  - ALLOW_MODEL_REQUESTS=Falseで本番APIへの誤リクエストを防止。

### ステップ6: Logfire等のInstrumentation追加(任意)
- **目的:** 開発・運用時のトレーシング・監視を強化。
- **手順:**
  1. logfire.configure()等を初期化時に追加。
  2. 必要に応じてinstrument=Trueでエージェント監視を有効化。

---

- **最重要ポイント:**
  - 「PydanticAI化しても、既存BDDテストが全てパスする=互換性・完成度が担保されている」ことを常に最優先とする。
  - 段階的に小さな単位でリファクタ→テスト→検証→次の段階へというサイクルを徹底。
  - テストがパスしない場合は、必ず原因を明確化し、仕様逸脱がないよう修正。

## 7. 今後の課題・検討事項
- PydanticAIのバージョン管理・API仕様変更への追従
- 既存TypedDict型の段階的廃止・Pydanticモデルへの統一可否
- サブクラス間の共通化・責務分離のさらなる最適化
- テスト・ドキュメントの自動生成・型安全化

## 8. 参考情報
- [PydanticAI公式ドキュメント](https://ai.pydantic.dev/)
- [Logfire公式ドキュメント](https://logfire.dev/)
- [Pydantic v2公式](https://docs.pydantic.dev/)

---
(本ファイルはPydanticAI導入に関する設計・議論・意思決定の記録として随時更新する)

## 型定義専用モジュール(types.py)新設と型管理方針

### 概要
- 2025-05-11頃、共通型(TypedDict, Pydanticモデル等)を一元管理するため、`src/image_annotator_lib/core/types.py` を新設。
- 当初は `WebApiComponents` などの基本的な型定義から開始。
- 型定義専用モジュールは「依存の最下層」に置き、他の自作モジュールに依存しない設計とする。
- これにより循環参照を防止し、保守性・型安全性・拡張性を最大化。
- `base.py`, `model_factory.py`, 各APIサブクラス等はすべて `types.py` から型をimportして利用する。

### 詳細設計・運用ルールと変更経緯
- **初期 (2025-05-11頃):**
    - `types.py` を作成し、Web API関連で共通的に利用できそうな型として `WebApiComponents` などを定義。
- **Pydanticモデル導入とインターフェース統一 (2025-05-06 - 2025-05-07頃):**
    1.  **`AnnotationSchema` の導入:** Web APIのレスポンス構造をPydanticモデル `AnnotationSchema(BaseModel)` として `types.py` に定義。これにより、APIレスポンスのバリデーションと型安全なデータアクセスを強化。
    2.  **`RawOutput` の変更:**
        - 当初 `_run_inference` の戻り値に含まれるアノテーションデータ (`response` キー) は `dict | None` であった。
        - `AnnotationSchema` 導入に伴い、`RawOutput` の `response` を `AnnotationSchema | None` に変更。これにより、`_run_inference` の段階でレスポンスがバリデーション済みPydanticモデルとして扱えるようになった。
    3.  **`WebApiFormattedOutput` の変更と `_format_predictions` の共通化:**
        - 当初、各Web APIサブクラスは独自の `_format_predictions` を持ち、戻り値の `annotation` 部分の型も統一されていなかった (Pydanticモデルを直接返すものもあった)。
        - ライブラリ全体 (ローカルモデル等、`BaseAnnotator` を継承する他のクラス) との出力形式の互換性を最優先とし、`BaseAnnotator.predict()` が返す最終結果 (`AnnotationResult`) の構造を維持するため、`_format_predictions` の `annotation` は辞書型 (`dict | None`) であるべきと判断。
        - これに伴い、`WebApiFormattedOutput` の `annotation` を `dict | None` に変更。
        - `WebApiBaseAnnotator` に `_format_predictions` メソッドを共通実装。このメソッド内で、`RawOutput` の `response` (`AnnotationSchema`) を `.model_dump()` を使って辞書に変換し、`WebApiFormattedOutput` の `annotation` に格納する処理を統一。
        - これにより、各Web APIサブクラスから `_format_predictions` の実装を削除し、基底クラスの共通処理を利用する形にリファクタリング。
    4.  **`WebApiInput` の導入:** Web APIへの入力(画像データ)を表現するPydanticモデル `WebApiInput(BaseModel)` を `types.py` に定義し、バリデーション(base64かbytesのどちらかが必須)を追加。
    5.  **影響:** これらの変更により、`_run_inference` (戻り値 `list[RawOutput]`) と `_format_predictions` (戻り値 `list[WebApiFormattedOutput]`) のインターフェースがWeb APIアノテーター間で統一され、型安全性が向上し、コードの重複が削減された。
- **現在の主要な型定義 (抜粋):**
    - `WebApiComponents(TypedDict)`: APIクライアント、モデルID、プロバイダー名を保持。
    - `WebApiInput(BaseModel)`: base64またはbytes形式の画像入力とバリデーション。
    - `AnnotationSchema(BaseModel)`: tags, captions, score を持つPydanticスキーマ。
    - `RawOutput(TypedDict)`: `response: AnnotationSchema | None`, `error: str | None` を持つ。
    - `WebApiFormattedOutput(TypedDict)`: `annotation: dict | None`, `error: str | None` を持つ。
- 型定義が膨大になる場合は、`types_webapi.py`, `types_core.py`等に細分化してもよいが、「依存の最下層」ルールは厳守。
- API固有の拡張型が必要な場合は、`types.py`の共通型を継承し、各APIサブモジュールで拡張。
- PydanticAI化後も、外部APIとのやりとりや型安全なデータ流通にはPydanticモデル、内部の辞書型構造や既存コード互換にはTypedDictを使い分ける。

### 循環参照防止の観点
- 型定義専用モジュールは「依存の最下層」に置くことで、どの実装ファイルからもimportでき、循環参照が絶対に発生しない。
- 型定義モジュールは他の自作モジュールに依存しないことが鉄則。
- 依存関係の流れ:types.py → base.py, model_factory.py, 各APIサブクラス → 上位ロジック

### 今後の型管理方針
- 型定義の重複・分散を防ぎ、どこからでもimportできるようにする。
- PydanticAI化やAPI追加時も型の一元管理が容易。
- テスト・型チェック・自動ドキュメント生成にも流用しやすい。
- 型定義の変更・追加時は必ずtypes.py(または分割型定義モジュール)を更新し、ドキュメントにも記録する。
- **`RawOutput` と `WebApiFormattedOutput` の定義は、最新の `types.py` と常に同期させること。**

---

## _format_predictionsの戻り値・WebApiFormattedOutputの型方針について {#format-predictions-policy}

### 決定事項(2025-05-12, 更新 2025-08-07)
- `_format_predictions` メソッドは **`WebApiBaseAnnotator` に共通実装**された。(2025-08-07頃 実施)
- 戻り値の型は **`list[WebApiFormattedOutput]`** で統一する。
- `WebApiFormattedOutput` の `annotation` キーには、`RawOutput` の `response` フィールド(`AnnotationSchema` オブジェクト)を **`.model_dump()` で変換した `dict` 型**のデータ、またはエラー時は `None` を格納する。(2025-05-07頃 方針決定・実装)
- Pydanticモデル(`AnnotationSchema`等)は主に `_run_inference` での**内部バリデーション・型安全化のために利用し、外部インターフェース (`_format_predictions` の戻り値) では `dict` へ変換して返す**。
- これは、API経由でないローカルモデルや従来型モデルも含め、ライブラリ全体の出力形式の一貫性・保守性・テスト互換性を最優先するため。

### 理由
- `BaseAnnotator` の抽象メソッド `_format_predictions` は多様なモデルに対応しており、戻り値の主要データ (`annotation`) を `dict` で統一することで全体の一貫性・保守性が高まる。
- 一部のアノテーターのみPydanticモデルで返すと、呼び出し側 (例: `BaseAnnotator.predict`) で型分岐や変換が必要となり、保守性・可読性が低下する。
- テスト・後続処理・既存コードとの互換性を維持しやすい。

### 今後の型移行方針
- 将来的に全モデルの出力形式をPydanticモデルで統一する場合は、`BaseAnnotator.predict` の戻り値型 (`AnnotationResult`) や関連する抽象クラス・全サブクラスの型アノテーションを一括で見直すタイミングで実施する。
- それまでは「`_run_inference` で Pydantic モデルを内部的に利用し、`_format_predictions` で `dict` に変換して返す」設計を維持する。

---

### _run_inference の戻り値型の統一について(2025-05-12追記, 更新 2025-05-07) {#run-inference-policy}
- `_run_inference` の戻り値は **`list[RawOutput]`** で統一する。(Web APIアノテーターについては2025-08-07頃 実施)
- `RawOutput` は `src/image_annotator_lib/core/types.py` で定義された `TypedDict` (`total=False`) であり、最新の定義は以下の通り。
    - `response: AnnotationSchema | None` # バリデーション済みPydanticモデル or None (2025-05-07頃 `dict | None` から変更)
    - `error: str | None`              # エラーメッセージ(なければNone)。2025-05-13の改修により、このエラーメッセージには原則として発生した例外のクラス名が含まれるようになった。
- すべてのWeb API系アノテーターでこの型に揃えることで、型安全・保守性・テスト互換性を担保する。(ローカルモデルアノテーターは現時点では異なる可能性があるため要確認 TODO:)
- テスト・実装・ドキュメントもこの型に揃えること。

--- 