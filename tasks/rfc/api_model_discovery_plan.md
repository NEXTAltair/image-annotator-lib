# API利用モデルの動的取得機能追加計画

## 1. 目的

*   `annotator_config.toml` で静的に定義されているAPIベースのアノテーター (Google, OpenAI, Anthropic, OpenRouter) が使用するモデル名を、各APIプロバイダーへの問い合わせによって動的に取得するように変更する。
*   特に画像を入力とするVisionタスクが実行可能なモデルに限定して取得する。
*   このモデル取得機能を外部から利用可能な関数として提供する。

## 2. 背景 / 現状の課題

*   現在、`annotator_config.toml` にAPIモデル名をハードコーディングしているため、プロバイダー側で新しいモデルが追加されたり、古いモデルが非推奨になっても自動で追従できない。
*   利用可能なモデルを手動で更新する必要があり、メンテナンスの手間がかかる。
*   ユーザーが利用したい最新のモデルが設定ファイルにないと、すぐには利用できない。

## 3. 方針

1.  **API仕様調査:** 各プロバイダー (Google Generative AI API, OpenAI API, Anthropic API, OpenRouter API) の公式ドキュメントやSDKを調査し、利用可能なモデル一覧を取得するAPIエンドポイント/メソッドとその仕様（認証方法、レスポンス形式など）を確認する。特に、Visionタスクが実行可能なモデルをどのように識別・フィルタリングできるか重点的に調査する。
2.  **実装場所:** モデル一覧取得ロジックを `src/image_annotator_lib/core/api_model_discovery.py` 内に実装する。(新規作成)
3.  **処理実装:** 上記ファイル内に、OpenRouter API を呼び出してモデル一覧を取得し、Vision モデルをフィルタリングする関数を実装する。適切なエラーハンドリング（ネットワークエラー、API エラー、レスポンス形式変更など）も行う。**エラーハンドリングには `src/image_annotator_lib/exceptions/errors.py` で定義された適切な例外クラス (`WebApiError` のサブクラスなど) を使用すること。**
4.  **外部公開関数の実装:** 引数を取らず、モデル取得を試みる公開関数 `discover_available_vision_models() -> dict[str, list[str] | str]` を実装する。戻り値は辞書で、キーは成功時は `"models"` (値はモデルIDのリスト)、失敗時は `"error"` (値はエラーメッセージ文字列) とする。この関数を `src/image_annotator_lib/__init__.py` 等で公開する。
5.  **キャッシュ/更新戦略 & TOML 保存:**
    *   **関数呼び出し時:** `discover_available_vision_models` 呼び出し時、`force_refresh=False` の場合は、まず `available_api_models.toml` ファイルを `load_available_api_models` で読み込む。ファイルが存在し内容があれば、その情報を基に結果を返す。
    *   **API 取得 (`force_refresh=True` 時):** `force_refresh=True` が指定された場合は、ローカルの `available_api_models.toml` ファイルの存在や内容に関わらず、**常に** OpenRouter API から最新のモデルリストを取得する。
    *   **API 取得 (`force_refresh=False` 時):** `force_refresh=False` で、かつ `load_available_api_models` が空の結果を返した場合（ファイルが存在しない、または空の場合）も、OpenRouter API から最新のモデルリストを取得する。
    *   **TOML 保存:** API から取得成功した場合、取得した Vision モデル情報を整形し (詳細は Point 8 を参照)、`src/image_annotator_lib/core/config.py` の `save_available_api_models` 関数を使用して `available_api_models.toml` ファイルに保存する。既存の TOML データ (`config.py` の `load_available_api_models` で読み込み) と比較し、`last_seen` の更新と、API から取得できなくなったモデルへの `deprecated_on` の追加を行う。
    *   **エラー発生時の対応:** API 呼び出しや TOML 読み書きに失敗した場合、TOML ファイルは更新しない。エラー情報を関数の戻り値で返す。
    *   **再取得不要:** 実行中の動的な再取得は行わない（関数の再呼び出しまでは TOML ファイルの内容が基になる）。
6.  **(任意) 既存コードへの統合:** 必要に応じて、既存のアノテータークラス (`GoogleApiAnnotator` など) やUI部分で、新しい `available_api_models.toml` ファイルまたは `discover_available_vision_models` 関数の結果を利用するように修正する (今回のスコープとするか要検討)。
7.  **設定ファイルの扱い:**
    *   `annotator_config.toml` (プロジェクトルート下 `config/`): ライブラリの基本的な設定ファイル。
        *   **初回生成:** このファイルがプロジェクトの `config/` ディレクトリに存在しない場合、`config.py` の `ModelConfigRegistry.load` が初回に呼び出された際に、パッケージ同梱のテンプレート (`src/.../annotator_config.toml`) から**自動的にコピー・生成される**。
        *   **理由:** `importlib.resources` で取得されるパッケージ内リソースパスは、インストール環境での書き込みが保証されないため、ユーザーが書き込み可能なプロジェクトルート下に設定ファイルを配置する必要がある。
    *   `available_api_models.toml` (プロジェクトルート下 `config/`): 動的に取得・更新されるモデル情報は、このファイルに保存・管理される。
        *   **初回生成:** このファイルが存在しない場合、`config.py` の `load_available_api_models` または `save_available_api_models` が初回に呼び出された際に**自動的に空ファイルまたはディレクトリが生成される**。
        *   **理由:** `annotator_config.toml` と同様の理由で、ユーザーが書き込み可能なプロジェクトルート下に配置する。
8.  **TOML 保存詳細 (`available_api_models.toml`):**
    *   **保存場所:** プロジェクトルート下の `config/available_api_models.toml` (`constants.py` でパスを定義)。
    *   **キー:** OpenRouter から取得したモデル ID (例: `openai/gpt-4o`)。
    *   **値 (辞書):**
        *   `provider` (str): `model['name']` または `model['id']` から抽出。
        *   `model_name_short` (str): `model['name']` または `model['id']` から抽出。
        *   `display_name` (str): 元の `model['name']` (例: "OpenAI: GPT-4o")。
        *   `created` (str): `model['created']` (Unix タイムスタンプ) を ISO 8601 形式 (UTC, 例: "2023-05-03T10:30:00Z") に変換したもの。
        *   `modality` (str): `model['architecture']['modality']` (例: "text+image->text")。
        *   `input_modalities` (list[str]): `model['architecture']['input_modalities']` (例: ["text", "image"])。
        *   `last_seen` (str): API から最後に取得成功した日時 (ISO 8601 形式 UTC)。関数実行時に更新。
        *   `deprecated_on` (str | None): API から取得できなくなった日時 (ISO 8601 形式 UTC)。取得できなくなったら設定、再度取得できたら `null` に戻す。
    *   **プロバイダー/モデル名分割:**
        1. `model['name']` を `": "` で分割試行。
        2. 失敗した場合、`model['id']` を `"/"` で分割試行 (プロバイダー名は大文字化などの整形を検討)。
        3. 両方失敗した場合、フォールバック処理 (例: `provider="Unknown"`, `model_name_short=name`)。
    *   **ライブラリ:** `toml` ライブラリを使用 (依存関係に既に存在)。

## 4. 非目標 (スコープ外)

*   モデル一覧取得以外のAPI機能の追加。
*   アノテータークラスの初期化ロジックの大幅な変更（今回のスコープではモデル取得関数の提供を主とし、統合は別途検討する）。

## 5. 実装フェーズとタスクチェックリスト

*   **フェーズ 1: 調査と設計**
    *   [X] 各APIプロバイダーのモデル一覧取得APIの仕様調査 (エンドポイント、認証、レスポンス、Visionモデルの識別方法) (OpenRouter 完了)
    *   [X] Visionモデルの具体的なフィルタリング基準の決定 (API情報 or キーワード)。
        *   **全てのプロバイダー共通:** OpenRouter API からモデルリストを取得後、`architecture.input_modalities` リストに `"image"` が含まれているモデルを抽出する。
    *   [X] `discover_available_vision_models` 関数の詳細設計 (エラーハンドリング詳細、戻り値の型確定)。
    *   [X] `available_api_models.toml` の扱い方針最終決定 (動的取得・更新用の新規ファイルとしてプロジェクトルート下の `config` ディレクトリに作成・管理する)。
*   **フェーズ 2: 実装**
    *   [X] `constants.py` で `SYSTEM_CONFIG_PATH` もプロジェクトルート下の `config/` を指すように変更。
    *   [X] `constants.py` にプロジェクトルート下の `config/available_api_models.toml` へのパス定義を追加。
    *   [X] `config.py` の `ModelConfigRegistry.load` に、`SYSTEM_CONFIG_PATH` が存在しない場合にテンプレートから自動コピーする機能を追加 (`importlib.resources` を一時的に使用)。
    *   [X] `api_model_discovery.py` 内に関数 (`discover_available_vision_models`) の基本ロジック実装。
    *   [X] OpenRouter API 呼び出し処理の実装。
    *   [X] Visionモデルのフィルタリングロジックの実装。
    *   [X] `model['name']` と `model['id']` を用いた分割処理の実装。
    *   [X] タイムスタンプ (created, last_seen, deprecated_on) の処理 (ISO 8601 変換) 実装。
    *   [X] `config.py` に `load_available_api_models` 関数の実装 (新しいパスを使用、ファイルなければ空を返す)。
    *   [X] `config.py` に `save_available_api_models` 関数の実装 (新しいパスを使用、ディレクトリなければ作成)。
    *   [X] `api_model_discovery.py` で `load_available_api_models` を呼び出す処理の実装。
    *   [X] `api_model_discovery.py` で `save_available_api_models` を呼び出す処理の実装。
    *   [X] エラーハンドリングの実装 (APIエラー、ネットワークエラー、TOML I/O エラー)。
    *   [X] `api_model_discovery.py` の Linter エラー修正 (型エラー、未定義変数など)。
    *   [X] 外部公開関数 `discover_available_vision_models` の実装と公開設定 (`__init__.py`)。
    *   [X] メモリキャッシュロジックを削除。
*   **フェーズ 3: テスト**
    *   [ ] `config.py` の `ModelConfigRegistry.load` の初回ファイルコピー機能のユニットテスト。
    *   [X] API 呼び出しとフィルタリングのユニットテスト (モックAPI使用)
        *   [X] OpenRouter (example/check_openrouter_models.py で確認)
    *   [X] `name`/`id` 分割処理のユニットテスト。
    *   [X] タイムスタンプ処理のユニットテスト。
    *   [X] `config.py` の `load/save_available_api_models` 関数のユニットテスト (新しいパス、モックファイルシステム使用)。
    *   [X] `api_model_discovery.py` の `deprecated_on` 更新ロジックのユニットテスト (モック TOML データ使用)。
    *   [X] エラーハンドリングのユニットテスト。
    *   [X] `discover_available_vision_models` 関数のBDDテスト (モック使用で主要な振る舞いをカバー)
*   **フェーズ 4: ドキュメント**
    *   [X] `discover_available_vision_models` 関数の Docstring 作成。
    *   [ ] 必要に応じて関連ドキュメント (README等) の更新。

## 6. リスクと対策

*   **リスク1:** API仕様の変更による機能停止。
    *   **対策:** エラーハンドリングを堅牢にし、APIエラー発生時はログ出力や空リスト返却などで対応。定期的な動作確認。
*   **リスク2:** Visionモデルの識別方法がプロバイダーによって異なる、または不明確。
    *   **対策:** OpenRouter API の `architecture.input_modalities` を使用する。
*   **リスク4:** OpenRouterのようなプロキシAPIの場合、モデルの能力（Vision対応か）をOpenRouter API自身が提供していない可能性。
    *   **対策:** `architecture.input_modalities` を確認済み。

## 7. 検討事項 (仕様確定待ち)

*   取得したモデルリストの最終的な利用方法 (関数公開のみ or 既存コード統合)