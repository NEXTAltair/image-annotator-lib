Version 1.1.0
# 変更計画

# 計画書・チェックリスト

## 1. 概要
image-annotator-lib は、画像アノテーション（タグ付け・スコアリング・キャプション生成など）を統一的なインターフェースで実行できる Python ライブラリです。

- ONNX, PyTorch (Transformers), TensorFlow, CLIP, Web API など多様なモデルタイプを抽象化し、共通の基底クラス (`BaseAnnotator`) とフレームワーク別基底クラスで管理します。
- モデル追加・拡張が容易な設計（3層クラス階層＋設定ファイル駆動）を採用し、開発者が最小限の実装で新規モデルを組み込めます。
- メモリ管理・キャッシュ・デバイス自動判定（CUDA/CPU）・バッチ推論・エラー処理・標準化された結果構造など、実運用を意識した堅牢な基盤を提供します。
- ドキュメント・ルール・設計方針は @docs, @memory-bank ディレクトリに記録・管理されており、設計変更時は必ずドキュメントを更新し整合性を保つ運用としています。

本計画書は、主にモデルロード時のCUDAエラーや設定不備によるTypeError等の実行時問題の解決、およびそれに伴う設計・実装・ドキュメントの修正を目的としています。

## 2. 変更点
本計画書で扱う主な変更・修正点は以下の通りです。

- CUDA未対応環境で `device="cuda"` が指定された場合の自動CPUフォールバック機能の実装
    - `core/utils.py:determine_effective_device` でCUDA利用可否を判定し、警告ログとともにCPUへ切り替え
    - `BaseAnnotator` など全モデルでこの判定を利用するよう統一
- モデル設定ファイル (`annotator_config.toml`) の `estimated_size_gb` 欠落による警告の解消
    - `aesthetic_shadow_v1` など該当モデルに `estimated_size_gb` を追加
- モデルロード失敗時のTypeError・キャッシュ解放エラーの根本解決
    - フォールバック・設定修正により、`Torch not compiled with CUDA enabled` や `NoneType` エラーを解消
- ドキュメント・ルール・設計記述の最新化
    - 設定ファイルパス・命名規則・型ヒント・Any型理由コメント・テスト構造など、@docs/@memory-bank の記録に基づき整合性を再確認
- これらの修正により、開発者・利用者が異なる環境でも安定してモデルを利用できるようにすることを目指します。

## 3. 現在の課題と対応計画

### 課題1: モデルロード時のCUDAエラーおよび関連TypeError
* **課題の説明:** `annotate` 実行時に `aesthetic_shadow_v1` モデルが `cuda` デバイスでロードできず、後続処理で `TypeError` が発生する問題。
    * **現象:**
        * `logs/image-annotator-lib.log` に以下のエラー・警告が出力されていた:
            * `WARNING: モデル 'aesthetic_shadow_v1' の estimated_size_gb が config に見つかりません。`
            * `ERROR: Pipeline 'aesthetic_shadow_v1' のロード中に予期せぬエラーが発生しました: Torch not compiled with CUDA enabled`
            * `ERROR: Pipeline 推論中にエラーが発生: 'NoneType' object is not subscriptable` (原因: モデルロード失敗)
            * `ERROR: モデル 'aesthetic_shadow_v1' のCPUへのキャッシュに失敗しました: 'NoneType' object has no attribute 'items'` (原因: モデルロード失敗)
    * **原因:**
        * インストールされている PyTorch が CUDA 非対応 (CPU版) であったこと。
        * モデル設定 (`aesthetic_shadow_v1`) に `estimated_size_gb` が欠落していたこと。
    * **解決策チェックリスト:**
        *   **ステップ 1: モデルサイズ設定の修正**
            *   [x] 設定ファイル (`src/image_annotator_lib/resources/system/annotator_config.toml`) を特定した。
            *   [x] `"aesthetic_shadow_v1"` の設定に `"estimated_size_gb": 0.6` を追加した。
            *   [x] コードを実行し、サイズ未設定の `WARNING` が消えることを確認した。
        *   **ステップ 2: CUDA 環境の確認と修正**
            *   [x] ユーザー環境で `torch.cuda.is_available()` が `False` であることを確認した。
            *   [x] **(修正案) CUDA利用不可時のCPUフォールバック:**
                *   [x] `torch.cuda.is_available()` が `False` かつ設定で `device="cuda"` が要求された場合に、警告ログを出力しCPUで実行することを通知するロジックを `core/utils.py` の `determine_effective_device` 関数として追加した。
                *   [x] `BaseAnnotator.__init__` 等で `determine_effective_device` を使用し、内部的にデバイスを `"cpu"` に切り替えて処理を続行するようにした。
                *   [x] コードを実行し、CUDA非対応環境で `device="cuda"` を指定してもCPUでロードが成功し、警告ログが表示されることを確認した。
            *   ~~[ ] **(CPU実行の場合)** モデルロード時のデバイス指定を `"cpu"` に変更する。~~ (フォールバック実装により不要)
            *   ~~[ ] **(CUDA実行の場合)** 必要に応じて CUDA 対応 PyTorch を再インストール、または CUDA 環境を修正する。~~ (フォールバック実装により不要)
            *   [x] コードを実行し、`Torch not compiled with CUDA enabled` エラーおよび関連する `TypeError` が解消されることを確認した。
        * **対応:** (2025-04-19)
            * `annotator_config.toml` に `aesthetic_shadow_v1` の `estimated_size_gb` を追加。
            * CUDA非対応環境で `device="cuda"` が指定された場合に、CPUへ自動的にフォールバックする機能を実装 (`core/utils.py:determine_effective_device` および関連箇所の修正)。
            * フォールバック時に警告ログを出力するようにした。
            * 上記対応により、CUDA非対応環境でも `aesthetic_shadow_v1` がCPUで正常にロード・実行され、後続の `TypeError` も解消されたことを確認。

## 4. 参考
- `docs/**/*`

## 5. チェックリスト
(タスクの内容を細分化したチェックリスト - 必要に応じて追加)


## 6. 改訂履歴
- 1.1.0 (2024-04-19): モデルロード時のCUDAエラーおよび関連TypeErrorの解決
- 1.0.0 (2024-04-19): 初版作成（他ファイルからの統合）
