# langmap.py
# Unified mapping from fastText ISO-639-1 codes to PaddleOCR language codes.
# Covers PaddleOCR PP-OCRv5 37 languages and falls back to English.
# Sources:
# - PaddleOCR PP-OCRv5 supported languages
# - FastText ISO-639-1 codes

FASTTEXT_TO_PADDLE = {
    # Arabic
    "ar": "ar",
    # Bengali
    "bn": "bn",
    # Bulgarian
    "bg": "bg",
    # Czech
    "cs": "cs",
    # Danish
    "da": "da",
    # Dutch
    "nl": "nl",
    # English
    "en": "en",
    # Estonian
    "et": "et",
    # Finnish
    "fi": "fi",
    # French
    "fr": "fr",
    # German
    "de": "de",
    # Greek
    "el": "en",
    # Hebrew
    "he": "en",
    # Hindi
    "hi": "hi",
    # Hungarian
    "hu": "hu",
    # Indonesian
    "id": "id",
    # Italian
    "it": "it",
    # Japanese
    "ja": "japan",
    # Korean
    "ko": "korean",
    # Latvian
    "lv": "lv",
    # Lithuanian
    "lt": "lt",
    # Malay
    "ms": "ms",
    # Norwegian
    "no": "no",
    # Polish
    "pl": "pl",
    # Portuguese
    "pt": "pt",
    # Romanian
    "ro": "ro",
    # Russian
    "ru": "ru",
    # Spanish
    "es": "es",
    # Swedish
    "sv": "sv",
    # Tamil
    "ta": "ta",
    # Telugu
    "te": "te",
    # Thai
    "th": "th",
    # Turkish
    "tr": "tr",
    # Ukrainian
    "uk": "uk",
    # Urdu
    "ur": "en",
    # Vietnamese
    "vi": "vn",
    # Chinese Simplified & Traditional
    "zh": "ch",
    "zh-cn": "ch",
    "zh-tw": "ch"
}

# Fallback for any code not explicitly listed
DEFAULT_PADDLE_LANG = "en"


def get_paddle_lang(ft_lang: str) -> str:
    """
    Convert a fastText ISO-639-1 code to the PaddleOCR model code.
    Returns DEFAULT_PADDLE_LANG for unsupported codes.
    """
    key = ft_lang.lower()
    return FASTTEXT_TO_PADDLE.get(key, DEFAULT_PADDLE_LANG)
