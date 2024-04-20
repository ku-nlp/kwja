from unicodedata import normalize

from kwja.utils.constants import TRANSLATION_TABLE


def normalize_text(text: str) -> str:
    return normalize("NFKC", text).translate(TRANSLATION_TABLE)
