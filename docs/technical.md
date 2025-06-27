# 技術仕様書

## 1. 開発環境

### 1.1 技術スタック

- Python >= 3.12
- PyTorch (Transformers, CLIP)
- ONNX Runtime
- TensorFlow (DeepDanbooru)
- TOML (設定ファイル)
- Ruff (フォーマッター、リンター)
- Mypy (型チェック)
- uv (パッケージ管理)
- pytest, pytest-xdist (テスト)
- Pydantic (データバリデーション、型安全性の向上)

### 1.2 主要依存関係

依存関係は `pyproject.toml` に定義、`uv` で管理。主要ライブラリは以下 (詳細は `pyproject.toml` 参照):

- `toml`
- `requests`
- `huggingface_hub`
- `transformers`
- `onnxruntime` / `onnxruntime-gpu`
- `tensorflow`
- `Pillow`
- `numpy`
- `tqdm`
- `pytest`, `pytest-cov`, `pytest-xdist` (テスト用)
- `psutil` (メモリ管理用)
- `python-dotenv` (APIキー管理用)
- `anthropic` (API用)
- `google-genai` (API用)
- `openai` (API用)
- `pydantic` (データバリデーション、型定義)
- `pydantic-ai` (PydanticモデルとLLMの連携)

### 1.3 ディレクトリ構造 (主要部分)

```mermaid
graph TD
    A[.] --> B(config)
    B --> B1(annotator_config.toml)
    A --> C(docs)
    C --> C1(architecture.md)
    C --> C2(product_requirement_docs.md)
    C --> C3(technical.md)
    A --> D(src)
    D --> D1(image_annotator_lib)
    D1 --> D1a(api.py)
    D1 --> D1b(core)
        D1b --> D1b1(config.py)
        D1b --> D1b2(model_factory.py)
        D1b --> D1b3(registry.py)
        D1b --> D1b4(types.py)
        D1b --> D1b5(provider_manager.py)  # New
        D1b --> D1b6(pydantic_ai_factory.py) # New
        D1b --> D1b7(base)
            D1b7 --> D1b7a(annotator.py)
            D1b7 --> D1b7b(transformers.py)
            D1b7 --> D1b7c(webapi.py)
    D1 --> D1c(model_class)
        D1c --> D1c1(annotator_webapi)
            D1c1 --> D1c1a(anthropic_api.py)
            D1c1 --> D1c1b(google_api.py)
            D1c1 --> D1c1c(openai_api_chat.py)
            D1c1 --> D1c1d(openai_api_response.py)
    A --> F(tests)
    F --> F1(integration)
    F --> F2(unit)
    A --> I(pyproject.toml)
```

## 2. コーディング規約

### 2.1 基本規約

- **言語:** 日本語 (エラーメッセージ、ログ、コメント、Docstring)
- **フォーマッター:** Ruff format (`ruff format .`)
- **リンター:** Ruff (`ruff check . --fix`)
- **型チェック:** Mypy (`mypy src/`)
- **Docstring:** Google スタイル (日本語)
- **ルール参照:** `.cursor/rules/rules.mdc` を最優先の規約とする。

### 2.2 特に重要なルール

- **型ヒント:**
    - **モダンな型:** `typing.List`, `typing.Dict`, `Optional` 等の古い型ではなく、Python 3.9+ の組込型 (`list`, `dict`) や `|` 演算子 (`str | None`) を使用。複雑な辞書は `TypedDict` を検討。
    - **データクラスとバリデーション:** Pydantic モデルを積極的に活用し、型安全性とバリデーションを強化する。
    - **オーバーライド:** 親クラスメソッドをオーバーライドする際は `@override` デコレーターを使用。
- **カプセル化:**
    - 他クラスの内部変数 (`_` 始まり) への直��アクセスは禁止。
    - Tell, Don't Ask の原則に従う。
    - 公開インターフェースは最小限にする (YAGNI)。

## 3. 主要技術決定 (履歴)

以下はプロジェクト開発中の主要な技術決定の記録。新しいものが上に来ます。

- **PydanticAI Provider-levelアーキテクチャの導入 (2025-06-25):**
    - **目的:** Web API利用の効率性、拡張性、保守性の向上。
    - **概要:** `ProviderManager` がAPIプロバイダー（Google, OpenAIなど）ごとにクライアントインスタンスを管理・共有し、`PydanticAIProviderFactory` が `pydantic-ai` の `Agent` を効率的にキャッシュ・再利用するアーキテクチャを導入。
    - **主要コンポーネント:**
        - `ProviderManager`: 推論実行の中央管理。
        - `PydanticAIProviderFactory`: `Agent` と `Provider` の生成とキャッシュ管理。
        - `PydanticAIAnnotatorMixin`: `pydantic-ai` を利用するアノテータの共通ロジック。
        - `PydanticAIWebAPIWrapper`: 既存APIとの後方互換性を保つためのラッパー。
    - **利点:** APIクライアント���再利用によるパフォーマンス向上とメモリ効率の改善。

- **型定義の一元管理 (`core/types.py`導入) (2025-05-13):**
    - **目的:** 循環参照を防止し、保守性を向上させるため、プロジェクト共通の型定義を `src/image_annotator_lib/core/types.py` に集約。

- **Web API アノテーターの責務分離 (2025-05-10):**
    - **目的:** 保守性と拡張性を高めるため、APIごとの実装を明確に分離。
    - **変更点:** `OpenAIApiAnnotator` と `AnthropicApiAnnotator` などを個別のファイルに分離し、`AnnotationSchema` を用いて出力の型仕様を統一。

- **CUDA非対応環境CPUフォールバック実装 (2025-04-19):**
    - **目的:** CUDAが利用できない環境でもライブラリが安定して動作するようにする。
    - **変更点:** 環境に応じて利用可能なデバイス(`cuda` or `cpu`)を自動判定し、CUDAが利用不可の場合はCPUにフォールバックする機能を実装。

- **ログ出力ライブラリの変更と初期化処理の改善 (2025-04-18):**
    - **目的:** ログ出力の柔軟性向上と、多重初期化問題の解決。
    - **変更点:** ロギングライブラリを標準の `logging` から `loguru` に変更し、初期化処理を改善。

## 4. 新モデル追加方法

### 4.1 ローカルモデルの追加

1.  **クラス作成**: `src/image_annotator_lib/model_class/` に、適切な基底クラス（例: `TransformersBaseAnnotator`）を継承した新しいモデルクラスを作成します。
2.  **メソッド実装**: モデル固有のロジック（主に `_generate_tags` や `_run_inference`）を実装します。
3.  **設定追加**: `config/annotator_config.toml` に新しいモデルのセクションを追加し、`class`、`model_path` などを指定します。

### 4.2 Web APIモデルの追加 (Provider-Levelアーキテクチャ)

1.  **クラス作成**: `src/image_annotator_lib/model_class/annotator_webapi/` に、`WebApiBaseAnnotator` と `PydanticAIAnnotatorMixin` の両方を継承した新しいアノテータクラスを作成します。
2.  **`__init__` 実装**: `WebApiBaseAnnotator` と `PydanticAIAnnotatorMixin` の `__init__` を両方呼び出します。
3.  **`__enter__` 実装**: `self._setup_agent()` ��呼び出して `pydantic-ai` の `Agent` をセットアップします。
4.  **`run_with_model` 実装**: `ProviderManager` から呼び出されるこのメソッドで、実際の推論処理を実装します。通常は `self._run_inference_with_model()` を内部で呼び出します。
5.  **`_run_inference` 実装**: `annotate()` APIからの呼び出し（デフォルトモデルでの実行）のために、`self.run_with_model()` に処理を委譲します。
6.  **`ProviderManager` への追加**: `core/provider_manager.py` の `ProviderManager` と、対応する `ProviderInstanceBase` のサブクラスに新しいプロバイダーのロジックを追加します。
7.  **設定追加**: `config/annotator_config.toml` にモデル設定を追加します。`api_model_id` にはAPIで実際に使用するモデルIDを指定します。APIキーは `.env` ファイルで管理します。

**設定例 (Web API):**
```toml
[google-gemini-1.5-pro]
class = "GoogleApiAnnotator"
model_path = "gemini-1.5-pro-latest" # これは主にレジストリ用
api_model_id = "gemini-1.5-pro-latest" # APIコールで実際に使うID
provider = "google" # ProviderManagerが使用
```

## 5. テスト

- **テストフレームワーク**: `pytest` と `pytest-xdist` を使用。
- **実行コマンド**: `uv run pytest tests/`
- **カバレッジ**: `uv run pytest --cov=src/image_annotator_lib tests/`
- **テストの種類**:
    - **ユニットテスト (`tests/unit/`)**: コンポーネントを個別にテスト。外部依存は `@patch` でモック化。
    - **統合テスト (`tests/integration/`)**: 複数コンポーネントを連携させてテスト。
- **BDDテスト**: 過去に存在したが現在は削除済み。再導入する場合は統合テストとして、モック等を使用しない方針。
