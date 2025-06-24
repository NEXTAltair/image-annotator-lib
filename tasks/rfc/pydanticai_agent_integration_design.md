# PydanticAI Agent 統合設計案 RFC

## 1. はじめに

本ドキュメントは、既存の `image-annotator-lib` における `WebApiBaseAnnotator` 及びそのサブクラス群（`OpenAIApiAnnotator`, `GoogleApiAnnotator`, `AnthropicApiAnnotator` 等）の機能を、PydanticAI の `Agent` を活用して再構築するための設計案を提示するものです。OpenRouterへの対応は現時点ではスコープ外とします。

## 2. 背景と目的

- **現状の課題:**
    - 各Web APIプロバイダごとに個別のリクエスト処理、レスポンス解析、エラーハンドリングロジックが実装されており、コードの重複やメンテナンスコストの増大が見られる。
    - プロンプトの管理や、画像とテキストを組み合わせたマルチモーダル入力の扱いが複雑化しやすい。
    - LLMからの応答の型安全性をより強固にしたい。
- **PydanticAI導入の目的:**
    - PydanticAIの `Agent` を中心としたアーキテクチャに移行することで、LLMとの対話ロジックを標準化し、コードの共通化と可読性向上を図る。
    - Pydanticモデルを活用し、リクエストとレスポンスの型安全性を強化する。
    - マルチモーダル入力（特に画像と指示テキスト）の取り扱いを簡素化し、柔軟性を高める。
    - 将来的な機能拡張（例: 複数のLLMツールを組み合わせた高度な処理）への対応を容易にする。

## 3. 設計方針

### 3.1. 主要コンポーネントと役割

1.  **`PydanticAIAgentAnnotator(WebApiBaseAnnotator)`:**
    *   `WebApiBaseAnnotator` を継承する新たな基底クラス。
    *   PydanticAI `Agent` のインスタンスを内部に保持し、実際のLLMとの通信はこの `Agent` を介して行う。
    *   `Agent` の初期化に必要な `LLMModel`（プロバイダごとのLLMクライアントラッパー）や、`Tool`（後述）の管理を行う。
    *   既存の `_run_inference` メソッド内で `Agent.run()` を呼び出す形に置き換える。
    *   画像データ（`bytes`）とプロンプトテキストを `Agent` が処理できる形式に変換して渡す責務を持つ。

2.  **プロバイダ別 `LLMModel` 実装:**
    *   PydanticAI が提供する `OpenAIModel`, `GeminiModel`, `AnthropicModel` などを活用する。
    *   APIキーはPydanticAIの仕組みに従い、環境変数から自動的に読み込まれることを期待する。
    *   `PydanticAIAgentAnnotator` の初期化時に、使用するプロバイダに応じた `LLMModel` インスタンスを生成し、`Agent` に設定する。

3.  **`ImageAnnotationTool(BaseTool)`:**
    *   PydanticAI の `BaseTool` を継承するクラス。
    *   特定の画像に対するアノテーション処理（例: 画像の内容説明、特定オブジェクトの検出など、プロンプトによって指示されるタスク）を実行する単一の「ツール」として機能する。
    *   入力としてプロンプトと画像データを受け取り、LLMからのアノテーション結果（構造化されたPydanticモデル）を返す。
    *   `PydanticAIAgentAnnotator` は、この `ImageAnnotationTool` を `Agent` に登録して使用する。

4.  **`AnnotationSchema(PydanticBaseModel)`:**
    *   既存の `core/types.py` で定義されている `AnnotationSchema` を、PydanticAI `Agent` の `output_type` として活用する。
    *   LLMからの応答はこのスキーマに基づいてパースされ、バリデーションされる。

### 3.2. 処理フロー

1.  **初期化 (`PydanticAIAgentAnnotator.__init__`):**
    *   モデル名 (`model_name`) と設定 (`config`) を受け取る。
    *   設定に基づき、対象プロバイダの `LLMModel`（例: `OpenAIModel(api_key=...)` 等）をインスタンス化する。APIキーはPydanticAIの標準的な方法で環境変数から取得される。
    *   `ImageAnnotationTool` をインスタンス化する。
    *   `Agent` を初期化し、`LLMModel` と `ImageAnnotationTool` を登録する。この際、`Agent` の `output_type` に `AnnotationSchema` を指定する。

2.  **画像アノテーション実行 (`PydanticAIAgentAnnotator.predict` -> `_run_inference`):**
    *   `_preprocess_images` で画像データを `bytes` 形式に変換（既存の仕組みを流用）。
    *   `webapi_shared.py` 等から取得したベースプロンプトと、処理対象の画像（`bytes`）を準備する。
    *   `Agent.run()` を呼び出す。
        *   `Agent` は内部で `ImageAnnotationTool` を呼び出し、プロンプトと画像データを渡す。
        *   `ImageAnnotationTool` は `LLMModel` を使用してLLMにリクエストを送信する。PydanticAIがマルチモーダル入力（画像とテキスト）を適切に処理する。
        *   LLMからの応答は `AnnotationSchema` に基づいてパース・バリデーションされる。
    *   `Agent.run()` の結果（パース済みの `AnnotationSchema` インスタンスまたはエラー情報）を `_run_inference` の戻り値とする。

3.  **結果のフォーマット (`PydanticAIAgentAnnotator._format_predictions`):**
    *   `_run_inference` から受け取った `AnnotationSchema` インスタンス（またはエラー）を、既存の `WebApiFormattedOutput` 形式に変換する。これは `BaseAnnotator._generate_result` で最終的に `AnnotationResult` に変換されるため、既存の出力形式との互換性を保つ。

### 3.3. エラーハンドリング

*   APIキー未設定、認証エラー、APIからのエラーレスポンス、タイムアウト、PydanticAI内部エラーなどは、`Agent.run()` の呼び出し周りで集約的に捕捉する。
*   捕捉したエラーは、`_run_inference` の戻り値である `RawOutput` (もしくはそれに類する型) の `error` フィールドに格納し、上位の `BaseAnnotator` のエラー処理フローに委ねる。
*   PydanticAIのバリデーションエラーも同様に扱う。

### 3.4. 設定とプロンプト

*   APIモデルID（例: "gpt-4o"）などの基本的な設定は、既存の `annotator_config.toml` と `ModelConfigRegistry` を引き続き利用する。
*   プロンプトテンプレートは、`webapi_shared.py` の `BASE_PROMPT` などを活用し、`Agent` に渡す際に画像データと組み合わせる。

### 3.5. 既存クラスの置き換え

*   `OpenAIApiAnnotator`, `GoogleApiAnnotator`, `AnthropicApiAnnotator` は、この新しい `PydanticAIAgentAnnotator` を継承し、主にプロバイダ固有の `LLMModel` の指定や、微細な設定調整を行うだけのシンプルなクラスになることを目指す。場合によっては、`PydanticAIAgentAnnotator` にプロバイダ名を渡すことで、内部で `LLMModel` を切り替える共通クラスとして一本化することも検討する。

## 4. 影響範囲と変更点

*   `model_class/annotator_webapi/` 配下の各プロバイダ別アノテータークラスが大幅に変更・簡素化される。
*   `core/base.py` の `WebApiBaseAnnotator` の役割が一部変更される可能性がある（PydanticAI `Agent` との連携を主眼とするため）。
*   `PydanticAI` ライブラリへの依存が新たに追加される。
*   テストコードは、この新しいアーキテクチャに合わせて修正が必要となる。

## 5. 検討事項・課題

*   **パフォーマンス:** PydanticAI `Agent` を介することによるオーバーヘッド。特に大量画像処理時の性能影響を注視する。
*   **エラー詳細度:** PydanticAIから返されるエラー情報が、既存のエラーハンドリング機構で十分に扱えるか。
*   **既存機能の網羅:** 現在の各Web APIアノテーターが持つ細かな機能（例: 特定のAPIパラメータ調整）を、PydanticAIの枠組みでどのように実現するか。`LLMModel` の初期化オプションや、`Agent.run()` 時の引数で対応可能か検討する。
*   **処理する画像の最大数:** デフォルトで8枚とのことだが、PydanticAIの `Agent` が一度に複数の画像を効率的に扱えるか、または `predict` ループ内で1枚ずつ `Agent` を呼び出す形になるか検討が必要。

## 6. 今後のステップ

1.  `PydanticAIAgentAnnotator` と `ImageAnnotationTool` のプロトタイプを実装する。
2.  まず1つのプロバイダ（例: OpenAI）で動作検証を行う。
3.  他のプロバイダ（Google, Anthropic）へ展開する。
4.  既存のテストを参考に、新しいアーキテクチャ用のテストを作成・実行する。
5.  パフォーマンス測定と最適化を行う。
6.  本RFCの内容を元に、関連ドキュメント (`architecture.md` 等) を更新する。

以上 