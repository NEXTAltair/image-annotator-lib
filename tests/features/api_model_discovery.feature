Feature: API経由での利用可能なVisionモデルの探索

  画像アノテーションライブラリの利用者として、
  外部API (OpenRouter) から利用可能なVisionモデルを発見したい。
  これにより、どのモデルがアノテーションに利用可能かを知ることができる。

Background:
  Given アプリケーションのキャッシュディレクトリが存在する

Scenario: APIからVisionモデルを正常に探索する
  Given OpenRouter APIが利用可能であり、Visionモデルを含む有効なモデルリストを返す
  When discover_available_vision_models 関数を呼び出す
  Then 利用可能なVisionモデルのリストを取得できる

Scenario: 強制リフレッシュはローカルファイルを無視してAPIから取得する
  Given 以前にOpenRouter APIが正常に呼び出され、結果がキャッシュされている
  And OpenRouter APIが利用可能であり、有効なモデルリストを返す
  When force_refreshをtrueにして discover_available_vision_models 関数を呼び出す
  Then APIから最新のVisionモデルのリストを再取得できる

Scenario: APIエラーを適切に処理する
  Given 以下のエラータイプがAPIで発生する:
    | error_type         |
    | Connection Timeout |
    | HTTP Error 500     |
    | Invalid JSON       |
    | Request Exception  |
  When discover_available_vision_models 関数を呼び出す
  Then モデルリストの取得に失敗したことがわかる
  And エラーの原因が各エラータイプで正しく判定されること

Scenario: 予期しないAPI応答形式を処理する
  Given OpenRouter APIが予期しない形式でデータを返す
  When discover_available_vision_models 関数を呼び出す
  Then モデルリストの取得に失敗したことがわかる
  And エラーの原因がAPI応答形式の問題であることがわかる