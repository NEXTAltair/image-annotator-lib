
from typing import TypedDict

from pydantic import BaseModel


class AnnotationSchema(BaseModel):
    """Google API の応答を JSONクラスインスタンスにする
    Googleの場合
    example:
        my_Google_Json_Schema:
            [Google_Json_Schema(tags=['1girl', 'facing front', ...], captions=['A red-haired or ...', '...'], score=8.75)]
    """
    tags: list[str]
    captions: list[str]
    score: float

class Responsedict(TypedDict, total=False):
    response: AnnotationSchema | None
    error: str | None

class FormattedOutput(BaseModel):
    annotation: AnnotationSchema | None
    error: str | None


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

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "captions": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
    "required": ["tags", "captions", "score"],
    #"propertyOrdering": ["tags", "captions", "score"],
}
