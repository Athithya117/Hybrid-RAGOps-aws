
"""
do_dynamic_tesseract_ocr.py

Runs OCR on an image using only the Tesseract languages
that FastText detects are present in the text.
"""

import os
import re
import sys
import logging

import cv2
import numpy as np
from PIL import Image
import pytesseract
import fasttext

# — Setup logging —
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# — Environment & model paths —
FASTTEXT_MODEL_PATH = os.getenv(
    "FASTTEXT_MODEL_PATH",
    "/usr/local/share/lid.176.bin"
)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# — Load FastText model —
try:
    ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    log.info(f"Loaded FastText model from {FASTTEXT_MODEL_PATH}")
except Exception as e:
    log.error(f"Failed to load FastText model: {e}")
    sys.exit(1)

# — Map FastText codes → Tesseract traineddata codes —
# ——— Mapping FastText codes → Tesseract traineddata codes ———
FASTTEXT_TO_TESSERACT = {
    # Latin‑script Western languages
    "af":  "afr",    # Afrikaans
    "als": "gsw",    # Swiss German
    "an":  "arg",    # Aragonese → (use 'arg' if you add it; else skip)
    "ast": "ast",    # Asturian
    "br":  "bre",    # Breton
    "ca":  "cat",    # Catalan
    "cs":  "ces",    # Czech
    "da":  "dan",    # Danish
    "de":  "deu",    # German
    "el":  "ell",    # Greek
    "en":  "eng",    # English
    "es":  "spa",    # Spanish
    "et":  "est",    # Estonian
    "eu":  "eus",    # Basque
    "fa":  "fas",    # Persian
    "fi":  "fin",    # Finnish
    "fr":  "fra",    # French
    "gl":  "glg",    # Galician
    "ga":  "gle",    # Irish
    "hr":  "hrv",    # Croatian
    "hu":  "hun",    # Hungarian
    "ia":  "ina",    # Interlingua
    "id":  "ind",    # Indonesian
    "is":  "isl",    # Icelandic
    "it":  "ita",    # Italian
    "la":  "lat",    # Latin
    "lb":  "ltz",    # Luxembourgish
    "lt":  "lit",    # Lithuanian
    "lv":  "lav",    # Latvian
    "nl":  "nld",    # Dutch
    "oc":  "oci",    # Occitan
    "pl":  "pol",    # Polish
    "pt":  "por",    # Portuguese
    "ro":  "ron",    # Romanian
    "ru":  "rus",    # Russian
    "sk":  "slk",    # Slovak
    "sl":  "slv",    # Slovenian
    "sv":  "swe",    # Swedish
    "sw":  "swa",    # Swahili
    "vi":  "vie",    # Vietnamese
    "yi":  "yid",    # Yiddish

    # Indic & South Asian
    "bn":  "ben",    # Bengali
    "gu":  "guj",    # Gujarati
    "hi":  "hin",    # Hindi
    "kn":  "kan",    # Kannada
    "ml":  "mal",    # Malayalam
    "mr":  "mar",    # Marathi
    "ne":  "nep",    # Nepali
    "or":  "ori",    # Oriya
    "pa":  "pan",    # Punjabi
    "ta":  "tam",    # Tamil
    "te":  "tel",    # Telugu
    "ur":  "urd",    # Urdu

    # East Asian
    "zh":  "chi_sim",   # Chinese (Simplified)
    # if you need Traditional Chinese, use 'chi_tra'
    "ja":  "jpn",       # Japanese
    "ko":  "kor",       # Korean

    # Cyrillic & Central Asian
    "az":       "aze",        # Azerbaijani (Latin)
    "az_cyrl": "aze_cyrl",    # Azerbaijani (Cyrillic) — custom if you add it
    "be":       "bel",        # Belarusian
    "bg":       "bul",        # Bulgarian
    "ka":       "kat",        # Georgian
    "kk":       "kaz",        # Kazakh
    "ky":       "kir",        # Kyrgyz
    "mk":       "mkd",        # Macedonian
    "mn":       "mon",        # Mongolian
    "sr":  "srp",            # Serbian (Cyrillic)
    "sr_latn": "srp_latn",    # Serbian (Latin)
    "uk":       "ukr",        # Ukrainian
    "uz":       "uzb",        # Uzbek (Latin)
    "uz_cyrl": "uzb_cyrl",    # Uzbek (Cyrillic)

    # Others / Minor languages (where traineddata exists)
    "amh": "amh",    # Amharic
    "asm": "asm",    # Assamese
    "aze": "aze",    # Azerbaijani
    "aze_cyrl": "aze_cyrl",
    "bod": "bod",    # Tibetan
    "bos": "bos",    # Bosnian
    "ceb": "ceb",    # Cebuano
    "ceb": "ceb",
    "ces": "ces",    # Czech (alias of slk)
    "cos": "cos",    # Corsican
    "cym": "cym",    # Welsh
    "dzo": "dzo",    # Dzongkha
    "eus": "eus",    # Basque (alias of eu)
    "fas": "fas",    # Persian (alias fa)
    "frk": "frk",    # Franconian (if installed)
    "gla": "gla",    # Scottish Gaelic
    "gle": "gle",    # Irish (alias ga)
    "hat": "hat",    # Haitian Creole
    "hif": "hif",    # Fiji Hindi
    "hrv": "hrv",    # Croatian
    "iku": "iku",    # Inuktitut
    "jav": "jav",    # Javanese
    "jpn": "jpn",    # Japanese (alias ja)
    "kat": "kat",    # Georgian (alias ka)
    "kaz": "kaz",    # Kazakh (alias kk)
    "khm": "khm",    # Khmer
    "kir": "kir",    # Kyrgyz (alias ky)
    "kmr": "kmr",    # Northern Kurdish
    "lao": "lao",    # Lao
    "lit": "lit",    # Lithuanian (alias lt)
    "mal": "mal",    # Malayalam (alias ml)
    "mar": "mar",    # Marathi (alias mr)
    "mya": "mya",    # Burmese
    "nor": "nor",    # Norwegian (alias nb)
    "nld": "nld",    # Dutch (alias nl)
    "oci": "oci",    # Occitan
    "pus": "pus",    # Pashto
    "san": "san",    # Sanskrit
    "sin": "sin",    # Sinhala
    "som": "som",    # Somali
    "sqi": "sqi",    # Albanian (alias sq)
    "swe": "swe",    # Swedish (alias sv)
    "tgk": "tgk",    # Tajik
    "tir": "tir",    # Tigrinya
    "tha": "tha",    # Thai
    "tlh": "tlh",    # Klingon (if installed)
    "tpi": "tpi",    # Tok Pisin
    "tur": "tur",    # Turkish (alias tr)
    "uig": "uig",    # Uyghur
    "vie": "vie",    # Vietnamese (alias vi)
    "yor": "yor",    # Yoruba
}


_CLEAN_RE = re.compile(r"[^\w\s]")

def detect_fasttext_lang(text: str) -> tuple[str, float]:
    """
    Return FastText language code and confidence for the given text.
    """
    cleaned = _CLEAN_RE.sub("", text).strip()
    if not cleaned:
        return "unk", 0.0
    label, score = ft_model.predict(cleaned, k=1)
    return label[0].replace("__label__", ""), score[0]

def get_valid_tesseract_langs(lines: list[str], min_conf: float = 0.7) -> str:
    """
    From a list of text lines, detect languages and
    return a '+'-joined string of Tesseract codes.
    Falls back to 'eng' if none meet the confidence threshold.
    """
    codes = set()
    for line in lines:
        lang, conf = detect_fasttext_lang(line)
        tess = FASTTEXT_TO_TESSERACT.get(lang)
        if tess and conf >= min_conf:
            codes.add(tess)

    if not codes:
        codes = {"eng"}

    # ensure installed languages only
    try:
        installed = set(pytesseract.get_languages(config=""))
        codes = codes.intersection(installed) or {"eng"}
    except Exception:
        # if get_languages fails, assume codes are installed
        pass

    return "+".join(sorted(codes))

def do_dynamic_tesseract_ocr(img: np.ndarray) -> tuple[list[str], str]:
    """
    Perform OCR on the image using a draft pass in English
    to detect which languages to enable, then rerun OCR
    with those languages. Returns (lines, langs_used).
    """
    # binarize for better recognition
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    pil = Image.fromarray(bi)

    # draft pass in English
    draft = pytesseract.image_to_string(pil, config="-l eng")
    draft_lines = [l.strip() for l in draft.splitlines() if l.strip()]

    # determine dynamic langs
    langs = get_valid_tesseract_langs(draft_lines)
    log.info(f"TESSERACT_LANG={langs}")

    # final pass with dynamic langs
    final = pytesseract.image_to_string(pil, config=f"-l {langs}")
    final_lines = [l.strip() for l in final.splitlines() if l.strip()]

    return final_lines, langs

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk into a BGR numpy array.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        log.error(f"Cannot read image: {path}")
        sys.exit(1)
    return img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: do_dynamic_tesseract_ocr.py <image_path>", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    img = load_image(img_path)
    lines, langs_used = do_dynamic_tesseract_ocr(img)

    # print chosen languages and OCR output
    print(f"[OCR_LANGS] {langs_used}")
    for line in lines:
        print(line)

