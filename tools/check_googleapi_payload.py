"""Google API の JSON Schema をテストするためのスクリプト。

動作確認済み

# 結果
response.text: [
  {
    "tags": [
      "1girl",
      "facing front",
      "red hair",
      "red eyes",
      "glasses",
      "black hat",
      "black veil",
      "red rose",
      "black dress",
      "rose on dress",
      "close-up",
      "anime style",
      "detailed hair",
      "smooth skin",
      "intricate lace",
      "pale complexion",
      "youthful appearance",
      "elegant",
      "gothic aesthetic"
    ],
    "captions": [
      "A red-haired anime girl with red eyes faces forward, wearing glasses, a black hat with a veil, and a black dress adorned with roses.",
      "The close-up shot highlights her elegant and gothic appearance."
    ],
    "score": 8.75
  }
]
my_Google_Json_Schema: [Google_Json_Schema(tags=['1girl', 'facing front', 'red hair', 'red eyes', 'glasses', 'black hat', 'black veil', 'red rose', 'black dress', 'rose on dress', 'close-up', 'anime style', 'detailed hair', 'smooth skin', 'intricate lace', 'pale complexion', 'youthful appearance', 'elegant', 'gothic aesthetic'], captions=['A red-haired anime girl with red eyes faces forward, wearing glasses, a black hat with a veil, and a black dress adorned with roses.', 'The close-up shot highlights her elegant and gothic appearance.'], score=8.75)]

"""

import os

from dotenv import load_dotenv
from google import genai
from PIL import Image
from pydantic import BaseModel

load_dotenv()

image = Image.open("tests/resources/img/1_img/file01.webp")

BASE_PROMPT = """As an AI assistant specializing in image analysis, analyze images with particular attention to:
                    Character Details (if present):

                    Facing direction (left, right, front, back, three-quarter view)

                    Action or pose (standing, sitting, walking, etc.)

                    Hand positions and gestures

                    Gaze direction

                    Clothing details from top to bottom

                    Composition Elements:

                    Main subject position

                    Background elements and their placement

                    Lighting direction and effects

                    Color scheme and contrast

                    Depth and perspective

                    Technical Aspects and Scoring (1.00 to 10.00):

                    Score images based on these criteria:

                    Technical Quality (0-3 points):

                    Image clarity and resolution

                    Line quality and consistency

                    Color balance and harmony

                    Composition (0-3 points):

                    Layout and framing

                    Use of space

                    Balance of elements

                    Artistic Merit (0-4 points):

                    Creativity and originality

                    Emotional impact

                    Detail and complexity

                    Style execution

                    Examples of scoring:

                    9.50-10.00: Exceptional quality in all aspects

                    8.50-9.49: Excellent quality with minor imperfections

                    7.50-8.49: Very good quality with some room for improvement

                    6.50-7.49: Good quality with notable areas for improvement

                    5.50-6.49: Average quality with significant room for improvement

                    Below 5.50: Below average quality with major issues

                    Format score as a decimal with exactly two decimal places (e.g., 7.25, 8.90, 6.75)

                    Provide annotations in this exact format only:

                    tags: [30-50 comma-separated words identifying the above elements, maintaining left/right distinction]

                    caption: [Single 1-2 sentence objective description, explicitly noting direction and positioning]

                    score: [Single decimal number between 1.00 and 10.00, using exactly two decimal places]

                    Important formatting rules:

                    Use exactly these three sections in this order: tags, caption, score

                    Format score as a decimal number with exactly two decimal places (e.g., 8.50)

                    Do not add any additional text or commentary

                    Do not add any text after the score

                    Use standard tag conventions without underscores (e.g., "blonde hair" not "blonde_hair")

                    Always specify left/right orientation for poses, gazes, and positioning

                    Be precise about viewing angles and directions

                    Example output:
                    tags: 1girl, facing right, three quarter view, blonde hair, blue eyes, school uniform, sitting, right hand holding pencil, left hand on desk, looking down at textbook, classroom, desk, study materials, natural lighting from left window, serious expression, detailed background, realistic style

                    caption: A young student faces right in three-quarter view, sitting at a desk with her right hand holding a pencil while her left hand rests on the desk, looking down intently at a textbook in a sunlit classroom.

                    score: 5.50
                """

SYSTEM_PROMPT = """
                    You are an AI that MUST output ONLY valid JSON, with no additional text, markdown formatting, or explanations.

                    Output Structure:
                    {
                        "tags": ["tag1", "tag2", "tag3", ...],  // List of tags describing image features (max 150 tokens)
                        "captions": ["caption1", "caption2", ...],  // List of short descriptions explaining the image content (max 75 tokens)
                        "score": 0.85  // Quality evaluation of the image (decimal value between 0.0 and 1.0)
                    }

                    Rules:
                    1. ONLY output the JSON object - no other text or formatting
                    2. DO NOT use markdown code blocks (```) or any other formatting
                    3. DO NOT include any explanations or comments
                    4. Always return complete, valid, parseable JSON
                    5. Include all required fields: tags, captions, and score
                    6. Never truncate or leave incomplete JSON
                    7. DO NOT add any leading or trailing whitespace or newlines
                    8. DO NOT start with any introductory text like "Here is the analysis:"

                    Example of EXACT expected output format:
                    {"tags":["1girl","red_hair"],"captions":["A girl with long red hair"],"score":0.95}
                """

# ここからした API プロバイダーにより処理が違う部分
from google.genai import types as google_types

api_key = os.getenv("GOOGLE_API_KEY")


def preprocess_images(images: list[Image.Image]) -> list[bytes]:
    """画像リストをバイトデータのリストに変換する"""
    from io import BytesIO

    encoded_images = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="WEBP")
        encoded_images.append(buffered.getvalue())
    return encoded_images


class Google_Json_Schema(BaseModel):
    tags: list[str]
    captions: list[str]
    score: float


for image_data in preprocess_images([image]):
    content = {
        "parts": [
            {"text": BASE_PROMPT},
            {"inline_data": {"mime_type": "image/webp", "data": image_data}},
        ],
        "role": "user",
    }

    max_output_tokens = 1000
    temperature = 0.8
    top_p = 0.95
    top_k = 40

    generation_config = google_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        temperature=temperature,
        response_schema=list[Google_Json_Schema],
        top_p=top_p,
        top_k=top_k,
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemma-3-1b-it",
        contents=content,
        config=generation_config,
    )
    # Use the response as a JSON string.
    print(f"response.text: {response.text}")

    # Use instantiated objects.
    my_Google_Json_Schema: list[Google_Json_Schema] = response.parsed
    print(f"my_Google_Json_Schema: {my_Google_Json_Schema}")
