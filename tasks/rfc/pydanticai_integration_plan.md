# PydanticAI導入検討･設計案

## 1. 検討背景･目的
- 画像アノテーションWebAPI系アノテーターの型安全性･保守性･拡張性向上
- 依存性注入･構造化出力･エージェント設計のモダン化
- Logfire等によるトレーシング･監視の容易化
- 既存API･クラス構造･テストとの互換性維持

## 2. 現状の課題
- 設定値･APIクライアントの管理が分散しがち
- APIレスポンスの型安全なバリデーションが弱い
- サブクラスごとの重複･責務分離が不十分
- テスト容易性･拡張性に課題

## 3. PydanticAI導入の設計方針
- 既存の`BaseAnnotator`/`WebApiBaseAnnotator`/サブクラスのAPI･インターフェースは維持しつつ、**`WebApiBaseAnnotator` およびそのサブクラスの内部実装をPydanticAI化する(依存性管理、レスポンス管理、型バリデーション)。ローカルモデル用アノテーターは現行アーキテクチャを維持する。**
- **依存性注入はPydanticAIの推奨形式を採用する。** 具体的には、APIキー、APIクライアントの初期化に必要な情報(ベースURL等)、あるいは初期化済みのAPIクライアントオブジェクトそのものをフィールドに持つPydanticモデルを定義し、これを `deps_type` として `Agent`(または直接的なPydanticAI機能利用時)に渡す。
- APIレスポンスをPydanticモデルでバリデーションし、既存TypedDict型に変換して返却 (または `AnnotationSchema` を直接利用)。
- 例外･エラー処理は既存カスタム例外を維持しつつ、必要に応じPydanticAIのエラーラッピングを追加
- Logfire等のInstrumentationは開発･運用時に任意で有効化

## 4. 主要変更点･設計例
### 4.1 依存性注入のPydanticモデル化

**方針変更 (2025-05-14): PydanticAI推奨形式の採用**

PydanticAIのドキュメントで推奨される形式に基づき、APIキー、クライアント初期化に必要な情報、または初期化済みクライアントオブジェクト自体を依存性モデルに含める方針とします。

```python
from pydantic import BaseModel, SecretStr
from httpx import AsyncClient # 例として httpx を使用

# 汎用的なAPIクライアント設定を含む基本依存性モデルの例
class BaseApiAgentDependencies(BaseModel):
    api_key: SecretStr # APIキーはSecretStrで扱うことを推奨
    # model_id: str # これはリクエストごとに変わる可能性もあるため、別途渡すか、固定ならここに含める
    timeout: int = 60
    # 他の共通設定(リトライ回数など)

# OpenAI API を利用する場合の依存性モデルの例
class OpenAIAgentDependencies(BaseApiAgentDependencies):
    client: AsyncClient | None = None # 初期化済みクライアントを渡す場合 (オプション)
    base_url: str | None = None # クライアントを内部で初期化する場合
    # model_id: str # OpenAIのモデルID (例: "gpt-4o")
    # temperature, max_tokens などのAPIパラメータは別途 core/types.py のモデルで定義し、
    # この依存性モデルにネストするか、実行時に別途渡すことを検討
    # (例: request_params: OpenAIChatApiParams)

# (types.py に定義済みの OpenAIChatApiDependencies なども参考に、
#  クライアントとリクエストパラメータをどのように組み合わせるか再検討)
```
**上記はあくまで概念例です。実際の依存性モデルは `src/image_annotator_lib/core/types.py` に定義されている既存の `XXXDependencies` モデルをベースに、以下の「依存性注入方針の詳細と採用経緯」で述べるPydanticAI推奨形式に合わせて調整します。**

### 4.2 レスポンスの型定義･RawOutputとWebApiFormattedOutputの利用 {#response-types}
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

## 5. トレードオフ･注意点
- 利点: 型安全･保守性･テスト容易性向上、重複削減、エラー処理一元化
- 注意点: モデル設計の煩雑化、TypedDict型との相互変換コスト、PydanticAIのAPI追従

## 6. 段階的導入案(BDDテスト互換性担保を最優先)

### ステップ1: 既存BDDテストの現状把握･動作確認
- [x] **目的:** 既存のWebApiBaseAnnotatorおよび各サブクラス(OpenAI/Anthropic/Google等)が、現状のBDDテストで仕様通りに動作しているかを正確に把握する。

- [x] **手順:**
  1. [x] **テスト資産の洗い出し**  
     `tests/features/`配下の全featureファイルとstep定義をリストアップし、どのAPI･モデル･ケース(正常/異常)がカバーされているか確認する。
  2. [x] **テスト実行準備**  
     必要なAPIキー･設定ファイル･サンプル画像など、テスト実行に必要な環境が揃っているか確認する。
  3. [x] **BDDテストの一括実行**  
     `pytest`等で全BDDテストを実行し、結果(パス/失敗/スキップ)を記録する。
  4. [x] **失敗時の詳細分析**  
     失敗したテストがあれば、エラー内容･再現手順･失敗原因(実装バグ/仕様逸脱/テスト不備等)を明確にし、現状の仕様との差分を整理する。
  5. [x] **現状仕様の明文化**  
     テスト結果･分析内容をもとに、現状の「仕様･制約･既知の問題点」を設計ドキュメント等に記録する。

- [x] **観点:**
  - [x] すべての主要API･モデル･異常系がテストでカバーされているか
  - [x] テストの再現性･安定性(外部APIの仕様変更やネットワーク障害等の影響も考慮)
  - [x] 失敗時の原因分析･記録が十分か

- [x] **記録例:**
  - [x] 2025-05-11: 全BDDテストがパス。OpenAI Visionのoutput_textパース方式修正により、APIレスポンス形式の揺れにも対応できることを確認。
  - [x] 2025-05-12: Google Gemini APIの空レスポンス時にテストがスキップされることを確認。仕様として許容する方針をドキュメント化。
  - [x] 2025-05-13:
    - [x] APIキー未設定時のBDDテスト(`test_apiキーが未設定の場合は認証エラーが発生する`)が当初`AssertionError`で失敗。原因は、`api.py`の`_handle_error`が生成するエラーメッセージに例外の型名(`ApiAuthenticationError`)が含まれていなかったため。`_handle_error`を修正し、エラーメッセージに例外型名を含めることでテストパスを確認。
    - [x] APIタイムアウト時およびAPIエラーレスポンス時のBDDテスト(`test_apiリクエストがタイムアウトした場合は適切なエラーを返す`, `test_apiからエラーレスポンスが返された場合は適切に処理する`)の安定性を向上。`webapi_annotate_steps.py`の関連するGivenステップを修正し、API呼び出しをモックして意図的なエラーを送出するように変更。また、Thenステップのアサーションを調整し、大文字･小文字を区別せずにエラーメッセージを検証するように修正。これにより、全BDDテストがパスすることを確認。

### ステップ2: 依存性･レスポンスのPydanticモデル定義
- [ ] **目的:** 既存の設定値･APIクライアント･レスポンス型をPydanticモデル/データクラスで明示的に定義し、型安全な設計に移行する準備。
- [ ] **手順:**
  1. [ ] まずOpenAI系(`src/image_annotator_lib/model_class/annotator_webapi/openai_api_chat.py`等)から着手。
  2. [ ] **依存性について、PydanticAI推奨形式(APIキー、クライアント初期化情報、または初期化済みクライアントをフィールドに持つPydanticモデル)に基づき、`core/types.py` の既存 `XXXDependencies` モデルを修正、またはこれを包含する新しい依存性モデルを定義する。** `model_factory.prepare_web_api_components` の機能は、この新しい依存性モデルを構築するためのヘルパーとして利用側で活用するか、依存性モデルの初期化ロジックに統合することを検討する。
  3. [x] レスポンス(タグ、キャプション、スコア等)もPydanticモデルで定義。 (`AnnotationSchema` として `core/types.py` に定義済み)
  4. [ ] 既存のTypedDict型との相互変換関数も用意。

### ステップ3: WebApiBaseAnnotatorの内部実装をPydanticAI化
- [ ] **目的:** 依存性注入･レスポンスバリデーション･ツール設計をPydanticAIベースにリファクタ。
- [ ] **手順:**
  1. [ ] 既存の`WebApiBaseAnnotator`の内部で、依存性(**ステップ2で定義したPydanticAI推奨形式のモデル**)･レスポンスをPydanticモデルで管理。(レスポンスは `RawOutput` / `WebApiFormattedOutput` と `AnnotationSchema` で管理)
  2. [x] `_format_predictions` メソッドを `WebApiBaseAnnotator` に実装し、`list[RawOutput]` を `list[WebApiFormattedOutput]` に変換する共通ロジックを集約。(2025-08-07頃 実施済み)
     - [x] この際、`RawOutput` の `response` (Pydanticモデル `AnnotationSchema`) を `.model_dump()` して `WebApiFormattedOutput` の `annotation` (辞書) に格納するようにした。
  3. [ ] 必要に応じてAgent/tool/system_prompt等のPydanticAI APIを導入。**その際、`deps_type` にはステップ2で定義した新しい依存性モデルを指定し、`run` メソッドでそのインスタンスを渡す。**
  4. [x] 外部インターフェース(predict()等)は既存のまま維持し、テスト互換性を保つ。

### ステップ4: サブクラス(OpenAI/Anthropic/Google等)の順次移行
- [x] **目的:** 各APIごとのサブクラスもPydanticAIベースに統一。
- [x] **手順:**
  1. [x] 各サブクラスから `_format_predictions` メソッドを削除し、基底クラスの実装を利用するように変更。(2025-08-07頃 実施済み)
  2. [x] 各サブクラスの `_run_inference` の戻り値を `list[RawOutput]` (responseが `AnnotationSchema | None`) に統一。(2025-08-07頃 実施済み)

### ステップ5: BDDテストによる互換性･完成度検証
- [x] **目的:** PydanticAI化後も既存BDDテストが全てパスすることを必ず確認し、仕様互換性･完成度を担保。
- [x] **手順:**
  1. [x] 各リファクタ段階ごとにBDDテストを必ず実行。
  2. [x] テストが失敗した場合は、原因を分析し、仕様逸脱がないよう修正。
  3. [x] すべてのテストがパスした時点で「互換性･完成度が担保された」と判断。
- [x] **補足:**
  - [x] pytestをテストハーネスとして利用。
  - [ ] PydanticAIのAgent.overrideやTestModel/FunctionModelを活用し、LLM呼び出しをモック化したユニットテストも併用可能。(現状コードからは確認できず)
  - [x] ALLOW_MODEL_REQUESTS=Falseで本番APIへの誤リクエストを防止。

### ステップ6: Logfire等のInstrumentation追加(任意)
- [ ] **目的:** 開発･運用時のトレーシング･監視を強化。
- [ ] **手順:**
  1. [ ] logfire.configure()等を初期化時に追加。
  2. [ ] 必要に応じてinstrument=Trueでエージェント監視を有効化。

---

- **最重要ポイント:**
  - 「PydanticAI化しても、既存BDDテストが全てパスする=互換性･完成度が担保されている」ことを常に最優先とする。
  - 段階的に小さな単位でリファクタ→テスト→検証→次の段階へというサイクルを徹底。
  - テストがパスしない場合は、必ず原因を明確化し、仕様逸脱がないよう修正。

## 7. 今後の課題･検討事項
- PydanticAIのバージョン管理･API仕様変更への追従
- 既存TypedDict型の段階的廃止･Pydanticモデルへの統一可否
- サブクラス間の共通化･責務分離のさらなる最適化
- テスト･ドキュメントの自動生成･型安全化

## 8. 参考情報
- [PydanticAI公式ドキュメント](https://ai.pydantic.dev/)
- [Logfire公式ドキュメント](https://logfire.dev/)
- [Pydantic v2公式](https://docs.pydantic.dev/)

---
(本ファイルはPydanticAI導入に関する設計･議論･意思決定の記録として随時更新する)

## 型定義専用モジュール(types.py)新設と型管理方針

### 概要
- 2025-05-11頃、共通型(TypedDict, Pydanticモデル等)を一元管理するため、`src/image_annotator_lib/core/types.py` を新設。
- 当初は `WebApiComponents` などの基本的な型定義から開始。
- 型定義専用モジュールは「依存の最下層」に置き、他の自作モジュールに依存しない設計とする。
- これにより循環参照を防止し、保守性･型安全性･拡張性を最大化。
- `base.py`, `model_factory.py`, 各APIサブクラス等はすべて `types.py` から型をimportして利用する。

### 詳細設計･運用ルールと変更経緯
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
- 型定義の重複･分散を防ぎ、どこからでもimportできるようにする。
- PydanticAI化やAPI追加時も型の一元管理が容易。
- テスト･型チェック･自動ドキュメント生成にも流用しやすい。
- 型定義の変更･追加時は必ずtypes.py(または分割型定義モジュール)を更新し、ドキュメントにも記録する。
- **`RawOutput` と `WebApiFormattedOutput` の定義は、最新の `types.py` と常に同期させること。**

---

## _format_predictionsの戻り値･WebApiFormattedOutputの型方針について {#format-predictions-policy}

### 決定事項(2025-05-12, 更新 2025-08-07)
- [x] `_format_predictions` メソッドは **`WebApiBaseAnnotator` に共通実装**された。(2025-08-07頃 実施)
- [x] 戻り値の型は **`list[WebApiFormattedOutput]`** で統一する。
- [x] `WebApiFormattedOutput` の `annotation` キーには、`RawOutput` の `response` フィールド(`AnnotationSchema` オブジェクト)を **`.model_dump()` で変換した `dict` 型**のデータ、またはエラー時は `None` を格納する。(2025-05-07頃 方針決定･実装)
- [x] Pydanticモデル(`AnnotationSchema`等)は主に `_run_inference` での**内部バリデーション･型安全化のために利用し、外部インターフェース (`_format_predictions` の戻り値) では `dict` へ変換して返す**。
- [x] これは、API経由でないローカルモデルや従来型モデルも含め、ライブラリ全体の出力形式の一貫性･保守性･テスト互換性を最優先するため。

### 理由
- `BaseAnnotator` の抽象メソッド `_format_predictions` は多様なモデルに対応しており、戻り値の主要データ (`annotation`) を `dict` で統一することで全体の一貫性･保守性が高まる。
- 一部のアノテーターのみPydanticモデルで返すと、呼び出し側 (例: `BaseAnnotator.predict`) で型分岐や変換が必要となり、保守性･可読性が低下する。
- テスト･後続処理･既存コードとの互換性を維持しやすい。

### 今後の型移行方針
- 将来的に全モデルの出力形式をPydanticモデルで統一する場合は、`BaseAnnotator.predict` の戻り値型 (`AnnotationResult`) や関連する抽象クラス･全サブクラスの型アノテーションを一括で見直すタイミングで実施する。
- それまでは「`_run_inference` で Pydantic モデルを内部的に利用し、`_format_predictions` で `dict` に変換して返す」設計を維持する。

---

### _run_inference の戻り値型の統一について(2025-05-12追記, 更新 2025-05-07) {#run-inference-policy}
- [x] `_run_inference` の戻り値は **`list[RawOutput]`** で統一する。(Web APIアノテーターについては2025-08-07頃 実施)
- [x] `RawOutput` は `src/image_annotator_lib/core/types.py` で定義された `TypedDict` (`total=False`) であり、最新の定義は以下の通り。
    - [x] `response: AnnotationSchema | None` # バリデーション済みPydanticモデル or None (2025-05-07頃 `dict | None` から変更)
    - [x] `error: str | None`              # エラーメッセージ(なければNone)。2025-05-13の改修により、このエラーメッセージには原則として発生した例外のクラス名が含まれるようになった。
- [x] すべてのWeb API系アノテーターでこの型に揃えることで、型安全･保守性･テスト互換性を担保する。(ローカルモデルアノテーターは現時点では異なる可能性があるため要確認 TODO:)
- [x] テスト･実装･ドキュメントもこの型に揃えること。

---

## 依存性注入方針の詳細と採用経緯 (YYYY-MM-DD更新)

### 決定事項
`WebApiBaseAnnotator` およびそのサブクラスにおける依存性注入は、PydanticAIのドキュメントで示される推奨形式を採用する。
具体的には、APIキー、APIクライアントの初期化に必要な情報(ベースURL等)、および/または初期化済みAPIクライアントオブジェクトをフィールドとして持つPydanticモデル(`core/types.py` の `XXXDependencies` を改修または新規作成)を定義し、これを `deps_type` としてPydanticAI `Agent`(または直接的なPydanticAI機能利用時)に設定し、実行時にそのインスタンスを注入する。

### 決定経緯
PydanticAI導入にあたり、依存性注入の具体的な方法として複数の案を検討した。

1.  **現行方式の踏襲:** `WebApiBaseAnnotator.__enter__` 内で `model_factory.prepare_web_api_components` を呼び出し、APIキー取得とクライアント初期化をアノテーター内部で行う。
2.  **ハイブリッド案:** `prepare_web_api_components` によるクライアント初期化は維持しつつ、その戻り値(クライアントオブジェクト)と、別途Pydanticモデルで定義したAPIリクエストパラメータ(温度設定など)を組み合わせた新しいPydanticモデルを、実行時に依存性として注入する。
3.  **PydanticAI推奨形式(今回採用):** APIキー、クライアント初期化に必要な情報、または初期化済みクライアントオブジェクトそのものをフィールドに持つPydantic依存性モデルを定義し、これを実行時に注入する。

当初、既存の便利なユーティリティ関数 `prepare_web_api_components` を最大限に活用するハイブリッド案も有力だった。しかし、以下の点を考慮し、PydanticAIのドキュメントで示される標準的なアプローチである「推奨形式」を採用することとした。

*   **シンプルさと標準への準拠:** PydanticAIの思想は、Pythonの既存のベストプラクティスに従い、シンプルで理解しやすい設計を目指すことにある。APIキーやクライアントのような重要な依存性を明示的に外部から注入する形式は、この思想に合致し、ライブラリのドキュメントやコミュニティの用例とも整合性が取りやすい。
*   **テスト容易性:** 依存性が外部から注入されることで、ユニットテスト時にモックオブジェクトを容易に差し替えることができ、テストの記述性と信頼性が向上する。
*   **柔軟性と制御:** APIクライアントのライフサイクルや設定(カスタムヘッダー、プロキシ設定など)を、アノテーターの利用側がより細かく制御できるようになる。
*   **ハイブリッド案の潜在的な複雑性:** ハイブリッド案は既存コードとの親和性が高いものの、依存性の解決フローが一部アノテーター内部に残り、結果としてPydanticAIの依存性注入の恩恵を最大限に活かせない可能性や、かえって処理フローが複雑になる懸念があった。

### 各方式の主な違い(APIキーとクライアントの扱い)

| 特徴         | 現行方式                                                                 | ハイブリッド案 (不採用)                                                                                                | PydanticAI推奨方式 (採用)                                                                                                                               |
| :----------- | :----------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| APIキー管理  | アノテーター内部 (`__enter__` で `prepare_web_api_components` 経由)         | アノテーター内部 (`prepare_web_api_components` 経由)                                                                 | 依存性モデルのフィールドとして定義 (例: `SecretStr`)。呼び出し側が設定。                                                                                     |
| クライアント初期化 | アノテーター内部 (`__enter__` で `prepare_web_api_components` 経由)         | アノテーター外部 (呼び出し側が `prepare_web_api_components` を使用) し、結果のクライアントオブジェクトを依存性の一部として注入 | 依存性モデルに初期化済みクライアントを含めるか、またはクライアント初期化に必要な情報(ベースURL等)を依存性モデルに含め、アノテーター(またはAgentのツール内)で初期化。 |
| 依存性注入   | `model_name` のみ。他は内部解決。                                            | `prepare_web_api_components` の戻り値とリクエストパラメータ用Pydanticモデルを組み合わせたものを注入。                                  | APIキー、クライアント関連情報、リクエストパラメータ等を含むPydanticモデルを注入。                                                                                 |

### `model_factory.prepare_web_api_components` の役割変更
PydanticAI推奨形式の採用に伴い、`model_factory.prepare_web_api_components` の主な役割は以下のいずれか、または組み合わせになる。

*   アノテーターの利用側(PydanticAI `Agent` の呼び出し側など)が、PydanticAIの依存性モデルインスタンスを構築するための**ヘルパー関数**として利用する。例えば、この関数でAPIキーの取得やプロバイダー名の特定を行い、その結果を依存性モデルのフィールドにセットする。
*   依存性モデルのファクトリメソッドや `model_validator` の中で、`prepare_web_api_components` の一部ロジックを再利用または参考にし、依存性モデル自身がクライアントを初期化する責務を持つようにする。

### 今後の作業への影響
この方針変更により、以下の作業が必要となる。
*   `core/types.py` の既存の `XXXDependencies` モデル群を、PydanticAI推奨形式(APIキー、クライアント関連情報を含む形)に修正、またはこれを包含する新しい依存性モデルを定義する。
*   `WebApiBaseAnnotator` およびそのサブクラスの `__init__`、`__enter__`(不要になる可能性あり)、`_run_inference` 等を、新しい依存性モデルを受け取り利用するように改修する。
*   `WebApiBaseAnnotator` のインスタンス化と実行を行う箇所のロジックを修正する(`prepare_web_api_components` の呼び出しと依存性モデルの構築･注入)。
*   関連するテストコードの修正。

---

## 型定義専用モジュール(types.py)新設と型管理方針

### 概要
- 2025-05-11頃、共通型(TypedDict, Pydanticモデル等)を一元管理するため、`src/image_annotator_lib/core/types.py` を新設。
- 当初は `WebApiComponents` などの基本的な型定義から開始。
- 型定義専用モジュールは「依存の最下層」に置き、他の自作モジュールに依存しない設計とする。
- これにより循環参照を防止し、保守性･型安全性･拡張性を最大化。
- `base.py`, `model_factory.py`, 各APIサブクラス等はすべて `types.py` から型をimportして利用する。

### 詳細設計･運用ルールと変更経緯
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
- 型定義の重複･分散を防ぎ、どこからでもimportできるようにする。
- PydanticAI化やAPI追加時も型の一元管理が容易。
- テスト･型チェック･自動ドキュメント生成にも流用しやすい。
- 型定義の変更･追加時は必ずtypes.py(または分割型定義モジュール)を更新し、ドキュメントにも記録する。
- **`RawOutput` と `WebApiFormattedOutput` の定義は、最新の `types.py` と常に同期させること。**

---

## _format_predictionsの戻り値･WebApiFormattedOutputの型方針について {#format-predictions-policy}

### 決定事項(2025-05-12, 更新 2025-08-07)
- [x] `_format_predictions` メソッドは **`WebApiBaseAnnotator` に共通実装**された。(2025-08-07頃 実施)
- [x] 戻り値の型は **`list[WebApiFormattedOutput]`** で統一する。
- [x] `WebApiFormattedOutput` の `annotation` キーには、`RawOutput` の `response` フィールド(`AnnotationSchema` オブジェクト)を **`.model_dump()` で変換した `dict` 型**のデータ、またはエラー時は `None` を格納する。(2025-05-07頃 方針決定･実装)
- [x] Pydanticモデル(`AnnotationSchema`等)は主に `_run_inference` での**内部バリデーション･型安全化のために利用し、外部インターフェース (`_format_predictions` の戻り値) では `dict` へ変換して返す**。
- [x] これは、API経由でないローカルモデルや従来型モデルも含め、ライブラリ全体の出力形式の一貫性･保守性･テスト互換性を最優先するため。

### 理由
- `BaseAnnotator` の抽象メソッド `_format_predictions` は多様なモデルに対応しており、戻り値の主要データ (`annotation`) を `dict` で統一することで全体の一貫性･保守性が高まる。
- 一部のアノテーターのみPydanticモデルで返すと、呼び出し側 (例: `BaseAnnotator.predict`) で型分岐や変換が必要となり、保守性･可読性が低下する。
- テスト･後続処理･既存コードとの互換性を維持しやすい。

### 今後の型移行方針
- 将来的に全モデルの出力形式をPydanticモデルで統一する場合は、`BaseAnnotator.predict` の戻り値型 (`AnnotationResult`) や関連する抽象クラス･全サブクラスの型アノテーションを一括で見直すタイミングで実施する。
- それまでは「`_run_inference` で Pydantic モデルを内部的に利用し、`_format_predictions` で `