from PIL import Image

from image_annotator_lib.model_class.annotator_webapi import (
    AnthropicApiAnnotator,
    GoogleApiAnnotator,
    OpenAIApiAnnotator,
    OpenRouterApiAnnotator,
)

# 画像を読み込み
image = Image.open("tests/resources/img/1_img/file01.webp")

# アノテーターを初期化
# with OpenRouterApiAnnotator("Qwen2.5 VL 32B Instruct (free)") as annotator:
#     # 単一画像に対する処理
#     results = annotator.predict([image], ["test_hash"])
#     print(results)
#     # 結果を表示
#     for result in results:
#         print(f"タグ: {result['tags']}")
#         print(f"エラー: {result['error']}")

# with AnthropicApiAnnotator("Claude 3.5 Sonnet") as annotator:
#     # 単一画像に対する処理
#     results = annotator.predict([image], ["test_hash"])
#     print(results)
#     # 結果を表示
#     for result in results:
#         print(f"タグ: {result['tags']}")
#         print(f"エラー: {result['error']}")

with GoogleApiAnnotator("Gemma 3 1B (free)") as annotator:
    # 単一画像に対する処理
    results = annotator.predict([image], ["test_hash"])
    print(results)
    # 結果を表示
    for result in results:
        print(f"タグ: {result['tags']}")
        print(f"エラー: {result['error']}")

# with OpenAIApiAnnotator("GPT-4.1 Nano") as annotator:
#     # 単一画像に対する処理
#     results = annotator.predict([image], ["test_hash"])
#     print(results)
#     # 結果を表示
#     for result in results:
#         print(f"タグ: {result['tags']}")
#         print(f"エラー: {result['error']}")
