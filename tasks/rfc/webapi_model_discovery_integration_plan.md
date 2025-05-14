# Web API アノテーターへの動的モデル発見機能の統合計画

## 1. 目的

`discover_available_vision_models` 関数によって動的に取得･更新される利用可能な Web API モデルの情報 (`available_api_models.toml`) を、`image-annotator-lib` の Web API ベースのアノテータークラス (`GoogleApiAnnotator`, `OpenAIApiAnnotator`, `AnthropicApiAnnotator`, `OpenRouterApiAnnotator`) の初期化プロセスに統合する。

## 2. 背景

*   現状、各 Web API アノテーターは、プロバイダー側で使用する実際のモデル名 (`model_name_on_provider`) を、クラス内のデフォルト値や設定ファイル (`annotator_config.toml`) で静的に決定している。
*   `api_model_discovery` 機能により、利用可能な Vision モデルとその詳細(プロバイダー上の正式な ID を含む)を動的に取得できるようになった。
*   この動的情報を活用することで、ライブラリが最新のモデルに追従しやすくなり、ユーザーが設定ファイルで指定するモデル名(例: `"Gemini 1.5 Pro"`) と、実際に API で使用されるモデル名(例: `"google/gemini-pro-1.5"`)の間のマッピングを自動化できる。

## 3. 方針

*   **ユーザーインターフェース:** ユーザーは `annotate` 関数などでモデルを指定する際、`available_api_models.toml` 内の `model_name_short` (例: `"Gemini 1.5 Pro"`) を使用する。
*   **初期モデル情報取得:**
    *   ライブラリ初期化時 (`registry.py` の `initialize_registry` など) に、まずプロジェクトルート下の `config/available_api_models.toml` の存在を確認する。
    *   **ファイルが存在しない場合:** `api_model_discovery._fetch_and_update_vision_models()` を呼び出して API から最新情報を取得し、`config/available_api_models.toml` を新規作成する (ネットワークアクセスが発生)。
    *   **ファイルが存在する場合:** API 呼び出しは行わず、既存の `config/available_api_models.toml` を読み込む。
*   **設定ファイルの自動更新 (`annotator_config.toml`):**
    *   上記で読み込んだ、または新規作成した `available_api_models.toml` のデータに基づいて実行する。
    *   `available_api_models.toml` 内の各 Web API モデルについて、`model_name_short` と `provider` を取得する。
    *   `annotator_config.toml` 内に `[model_name_short]` セクションが存在しない場合のみ、そのセクションを自動作成する。
    *   自動作成するセクションには、`provider` 文字列がクラス名に含まれるかで特定した正しい `class` 名 (例: `class = "GoogleApiAnnotator"`) と、固定の `max_output_tokens = 1800` を書き込む。一致するクラスが見つからない場合は `OpenRouterApiAnnotator` を使用する。
    *   既に存在するセクションやその中の既存設定は変更･上書きしない。
*   **モデル名解決とインスタンス化 (`ModelFactory`):**
    *   `annotate` 関数などから `model_name_short` を受け取る。
    *   `available_api_models.toml` を検索し、値の辞書内の `model_name_short` が渡された名前と一致するエントリを探す。
    *   **見つからない場合:** 警告ログを出力し、そのモデルの処理をスキップする (結果にエラー情報を含める)。
    *   **見つかった場合:**
        *   エントリの **キー** (例: `"google/gemini-pro-1.5"`) を `model_id_on_provider` として取得する。
        *   `annotator_config.toml` から `[model_name_short]` セクションを読み込み、`class` 名と、ユーザーが設定した可能性のある他の API パラメータ (`temperature`, `timeout` 等) を取得する。
        *   特定した `class` 名を使ってアノテーターをインスタンス化する。この際、`model_name=model_name_short` と `model_id_on_provider=model_id_on_provider` を引数として渡す。
*   **アノテータークラスの修正 (`WebApiBaseAnnotator` サブクラス):**
    *   `__init__` メソッドを変更し、`model_name` (`model_name_short`) と `model_id_on_provider` (プロバイダーAPIで使う実際のモデルID) を引数として受け取り、インスタンス変数として保持する。
    *   コンストラクタ内で `config_registry.get(self.model_name, "parameter_name", default_value)` を使用して、`max_tokens` (デフォルト1800), `temperature`, `timeout` などの API パラメータを `annotator_config.toml` から読み込む。
    *   `_run_inference` メソッド内で API を呼び出す際には、インスタンス変数に保持した `self.model_id_on_provider` を使用する。
    *   **重要 (Google):** `GoogleApiAnnotator` を実装または修正する際は、**必ず Google AI Python SDK (`google-genai` パッケージ) を使用してください。** 公式ドキュメントに従い、**`from google import genai` および `from google.genai import types` を使用すること。** **絶対に `google-generativeai` パッケージを使用してはいけません。これらは異なるパッケージであり、このプロジェクトでは `google-genai` を使用します。** 過去のバージョンのドキュメントや実装例に見られる `from google.generativeai` は現在の推奨パスではありません。また、SDK のドキュメントには初期リリース版に関する注意書き(本番環境での使用非推奨など)が含まれていたため、利用する SDK のバージョンとドキュメントをよく確認すること。
    *   **重要 (OpenAI):** `OpenAIApiAnnotator` や `OpenRouterApiAnnotator` を実装または修正する際は、**OpenAI Python ライブラリの公式ドキュメントに従い、`from openai import OpenAI` を使用してクライアントを初期化すること (`client = OpenAI()`)。** 以前の `import openai` のみを使用するスタイルは非推奨です。
*   **設定ファイルの扱い:**
    *   `annotator_config.toml` の Web API モデルセクションにあった従来の `model_name_on_provider` キーは**削除**する。
    *   API パラメータ (`max_tokens`, `temperature`, `timeout` など) は、`annotator_config.toml` の `[model_name_short]` セクションでユーザーが設定･調整できるようにする。
    *   `class` は `annotator_config.toml` の `[model_name_short]` セクションで管理される (初期化時に自動追加される)。

## 4. 実装フェーズとタスクチェックリスト

*   **フェーズ 1: 設計詳細化**
    *   [X] ユーザー指定名 (`model_name_short`) からプロバイダーモデルID (`model_id_on_provider`, TOMLキー) への解決方法を決定。
    *   [X] モデルが見つからない場合のフォールバック戦略を決定 (警告ログ + スキップ)。
    *   [X] `annotator_config.toml` の `model_name_on_provider` を削除し、APIパラメータは `[model_name_short]` セクションで管理する方針を決定。
    *   [X] `annotator_config.toml` の自動更新ロジック (`[model_name_short]` セクションと `class`, `max_output_tokens` の追加) を決定。
    *   [X] 初期モデル情報取得方法を決定 (初回 or `available_api_models.toml` なしの場合にAPI取得)。
    *   [X] `provider` 文字列からクラス名へのマッピング方法を決定。
        *   **モデルID (TOMLキー) に `:` が含まれる場合 (例: `google/gemma-3-27b-it:free`) は `OpenRouterApiAnnotator` を使用する。**
        *   それ以外の場合、`provider` 文字列がクラス名に含まれるかで判断し、なければ `OpenRouterApiAnnotator` をフォールバックとして使用する。
    *   [X] Web API アノテーターの初期化方法 (`__init__` は `model_name` のみ) を決定 (7.1)。
    *   [X] API コール時のモデル ID 解決･加工方法 (`__enter__` で実施) を決定 (7.2)。
    *   [X] `WebApiComponents` の内容 (`client`, `api_model_id`, `provider_name`) を決定 (7.3)。
*   **フェーズ 2: 実装**
    *   [ ] `registry.py` の初期化処理における無駄な処理を特定し、修正する。 # 最優先タスク
    *   [X] `config.py` の `load_available_api_models` が期待通り動作するか再確認。
    *   [X] `config.py` に `annotator_config.toml` を**編集･保存**する機能を追加する。既存セクションを上書きせず、指定セクションにキーと値を追加できる必要がある (**`config_registry.add_default_setting(section, key, value)`** として実装)。
    *   [ ] `registry.py` の `initialize_registry` (またはその呼び出し元) に、初期モデル情報取得と `annotator_config.toml` の自動更新処理を実装する。
        *   [ ] `config/available_api_models.toml` の存在を確認。
        *   [ ] 存在しない場合、`api_model_discovery._fetch_and_update_vision_models()` を実行。
        *   [ ] `load_available_api_models` を呼び出す。
        *   [ ] `available_api_models.toml` の各 Web API モデルについてループ。
        *   [ ] `model_name_short` と `provider` を取得。
        *   [ ] `registry.py` で利用可能なクラスリストを取得し、`provider` 文字列を含むクラス名を検索 (なければ `OpenRouterApiAnnotator`)。
        *   [ ] `annotator_config.toml` を読み込み、`[model_name_short]` セクションが存在しないか確認。
        *   [ ] 存在しない場合、実装された **`add_default_setting`** を使って `class` と `max_output_tokens` を書き込む。
    *   [X] 各 Web API アノテーター (`WebApiBaseAnnotator` を含む) の `__init__` を修正し、`model_id_on_provider` 引数を削除。`model_name` (`model_name_short`) のみを受け取るようにする。
    *   [X] `ModelFactory` に Web API コンポーネント準備関数 (`prepare_web_api_components`) を実装。
        *   [X] `model_name` を引数に取る。
        *   [X] `available_api_models.toml` をロード。
        *   [X] `model_name` をキーに TOML データを検索。
        *   [X] `provider_name` と `model_id_on_provider` を取得。
        *   [X] 環境変数から `api_key` をロード。
        *   [X] `api_model_id` を加工。
        *   [X] API クライアントを初期化。
        *   [X] `WebApiComponents` を返す。
    *   [ ] `src/image_annotator_lib/resources/system/annotator_config.toml` から Web API モデルの `model_name_on_provider` を削除する。
    *   [ ] `OpenAIApiAnnotator` の 400 Bad Request エラー対応 (サブタスク):
        *   [ ] OpenAI の `responses.create` API が現在非推奨または廃止されていないか、対象モデルで利用可能かを確認する (ドキュメント確認 or 最小限のリクエスト試行)。
        *   [ ] もし `responses.create` が 400 エラーの原因であれば、`OpenAIApiAnnotator._run_inference` を `chat.completions.create` を使用する実装に戻す。
        *   [ ] `OpenAIApiAnnotator._format_predictions` を `chat.completions.create` の応答形式 (ChatCompletion オブジェクト) を正しく処理するように戻す/修正する。
        *   [ ] 以前 `chat.completions.create` 使用時に発生した `"'message'"` エラーの原因を特定し、修正する (応答形式のデバッグ、エラーハンドリングの確認など)。
        *   [X] `api.py` の `_create_annotator_instance` から OpenAI プレフィックス除去ロジックを削除する。
        *   [X] `base.py` の `WebApiBaseAnnotator` に `_get_processed_model_id` メソッドを追加し、プロバイダーに応じたプレフィックス除去処理を実装する。
        *   [X] 各 Web API サブクラス (`Google`, `OpenAI`, `Anthropic`) の `_run_inference` で `_get_processed_model_id` を呼び出し、加工済み ID を API コールに使用するように修正する。
        *   [ ] `example_lib.py` で `OpenAIApiAnnotator` (および他の Web API アノテーター) を使用し、400 エラーが発生せず、期待通りに動作することを確認する。
*   **フェーズ 3: テスト**
    *   [X] `config.py` の新しい設定書き込み機能のユニットテスト。
    *   [ ] `registry.py` の初期化処理が `available_api_models.toml` なしの場合に API を呼び出し、`annotator_config.toml` を正しく更新するかのユニットテスト (モック使用)。
    *   [ ] `registry.py` の初期化処理が `available_api_models.toml` ありの場合に API を呼び出さず、`annotator_config.toml` を正しく更新するかのユニットテスト (モック使用)。
    *   [ ] `registry.py` の初期化処理が `provider` からクラス名を正しくマッピングできるかのテスト。
    *   [ ] `ModelFactory` が `model_name_short` から正しい `model_id_on_provider` を解決し、アノテーターに渡すかのユニットテスト (モック使用)。
    *   [ ] Web API アノテーターが渡された `model_id_on_provider` を使って API コールを行うかのユニットテスト (モック使用)。
    *   [ ] `annotate` 関数を用いたエンドツーエンドテストで、Web API モデルが動的なモデル名解決を経て正常に動作することを確認する (必要ならモック API を使用)。
    *   [ ] TOMLにモデル情報がない場合に、対象モデルがスキップされることのテスト。
*   **フェーズ 4: ドキュメント**
    *   [ ] `developer_guide.md` や `api.md` を更新し、Web API モデルの利用方法 (`model_name_short` を使うこと) や設定に関する記述を修正する (`available_api_models.toml` の役割、`annotator_config.toml` の自動更新とパラメータ設定について強調)。
    *   [ ] `annotator_config.toml` のサンプルや説明を更新する (`model_name_on_provider` 削除、`[model_name_short]` セクションについて言及)。

## 5. 非目標 (スコープ外)

*   `api_model_discovery.py` 自体の機能変更。
*   Web API 以外のアノテータークラスへの変更。
*   ライブラリを利用するアプリケーション側の UI 変更。

## 6. リスクと対策

*   **リスク1:** `available_api_models.toml` が存在しない、古い、または破損している場合、モデル名解決に失敗または誤ったモデルが使用される。
    *   **対策:** フォールバック処理 (警告ログ + スキップ) を明確に定義する。ドキュメントで `discover_available_vision_models` の定期的な実行や `force_refresh=True` の利用を推奨する。
*   **リスク2:** `ModelFactory` の修正が他のモデルタイプのロードに影響を与える。
    *   **対策:** 修正箇所を Web API アノテーターの処理に限定し、既存のテストスイートでリグレッションがないことを確認する。
*   **リスク3:** `annotator_config.toml` の自動更新処理が、ユーザーの手動変更と競合したり、予期せぬ上書きを行う。
    *   **対策:** 更新処理はセクションが存在しない場合のみ追加するように限定し、既存の設定は変更しないことを徹底する。処理内容をログで明確に出力する。
*   **リスク4:** `model_name_short` が異なるプロバイダ間で重複する(ユーザー確認済みだが念のため)。
    *   **対策:** `available_api_models.toml` の検索時に `model_name_short` が複数見つかった場合の処理(警告ログ、エラー、最初のものを使うなど)を検討･実装する。
*   **リスク5:** 初回起動時にネットワーク接続がない、または API サーバーがダウンしている場合、`available_api_models.toml` が作成されず、Web API モデルが利用できない。
    *   **対策:** エラーハンドリングを実装し、問題をログに出力する。ドキュメントで初回起動時のネットワーク要件を明記する。
*   **リスク6:** `provider` 文字列を含むクラス名が意図せず複数存在する場合、間違ったクラスが選択される可能性がある。
    *   **対策:** `registry.py` でクラスを収集する際に、より厳密な命名規則を設けるか、マッピング方法を見直す (将来的検討)。現状は警告ログなどで対応。

## 7. 設計に関する追加検討 (2024-04-22)

### 7.1. Web API アノテーターの初期化方法

**検討事項:**
Web API アノテーター (`GoogleApiAnnotator` など) の `__init__` メソッドが、ユーザー指定のモデル名 (`model_name`, 実質 `model_name_short`) のみを受け取るようにし、ローカルモデルのアノテーターとインターフェースを統一すべきか検討した。

**決定:**
**`__init__` は `model_name` のみを受け取る**ように変更する。

**理由:**
ユーザーの視点から見て、アノテーターの初期化インターフェースがローカルモデルと統一され、分かりやすくなる。Web API モデルとローカルモデルの本質的な違い(ID解決の必要性)は、`__enter__` メソッド内で吸収する。

### 7.2. API コール時のモデル ID 解決･加工方法

**検討事項:**
各 Web API クライアント (Google, OpenAI, Anthropic など) の API 呼び出し時に `model` 引数として渡す最終的なモデル ID を、どのように準備･決定するのが最も分かりやすいか検討した。

**決定:**
**`__enter__` メソッド内で ID 解決と加工を行う**ように変更する。
1.  `__enter__` 内で `available_api_models.toml` を読み込む。
2.  `self.model_name` を基に `model_id_on_provider` を特定する。
3.  特定した `model_id_on_provider` をプロバイダー固有のルールで加工し、最終的な API 用 ID (`api_model_id`) を生成･保持する。
4.  `_run_inference` は保持された `api_model_id` を直接使用する。
5.  従来の `_get_processed_model_id` メソッドは削除する。

**理由:**
- リソース準備を行う `__enter__` で ID 解決と API クライアント初期化をまとめて行うのが自然である。
- `_run_inference` が ID 加工の詳細を意識する必要がなくなり、シンプルになる。
- `ModelFactory` が ID 加工ルールを知る必要がなく、責務分担が明確になる。

### 7.3. `WebApiComponents` 内容の定義

**検討事項:**
`__enter__` メソッドで生成･保持する `WebApiComponents` (TypedDict) に何を含めるべきか検討した。

**決定:**
`WebApiComponents` には以下を含める。
- `client: Any`: 初期化された API クライアント。
- `api_model_id: str`: API コールに使用する最終的な加工済みモデル ID。
- `provider_name: str`: プロバイダー名 (例: "Google")。
API パラメータ (temperature, timeout など) は含めない。

**理由:**
- `client` と `api_model_id` は `__enter__` で準備される主要な要素である。
- `provider_name` はコンポーネントセットの識別情報として有用である。
    - サブクラスの `self.provider_name` と冗長になる可能性はあるが、利用箇所では `self.components["provider_name"]` を参照することを推奨。
- API パラメータは `config_registry` が管理する責務であり、`WebApiComponents` に含めると責務が混同し冗長になるため含めない。パラメータは必要な時に `config_registry.get(...)` で取得する。

## 8. 改訂履歴
- 1.4.0 (2024-04-22) Web API アノテーターの初期化方法
- 1.3.0 (YYYY-MM-DD): providerからクラス名へのマッピング方法を確定。
- 1.2.0 (YYYY-MM-DD): available_api_models.toml がない場合に API 取得する仕様を追記。
- 1.1.0 (2025-04-20T16:18): ユーザーフィードバックに基づき、annotator_config.toml の自動更新方針とモデル名解決プロセスを修正。
- 1.0.0 (YYYY-MM-DD): 初版作成