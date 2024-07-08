from unicodedata import normalize

from rhoknp import Morpheme

from kwja.utils.constants import TRANSLATION_TABLE


def normalize_text(text: str) -> str:
    return normalize("NFKC", text).translate(TRANSLATION_TABLE)


def normalize_morpheme(morpheme: Morpheme) -> None:
    morpheme.text = normalize_text(morpheme.text)
    morpheme.reading = normalize_text(morpheme.reading.replace("う゛", "ゔ"))
    morpheme.lemma = normalize_text(morpheme.lemma)
    canon = morpheme.semantics.get("代表表記")
    if isinstance(canon, str):
        morpheme.semantics["代表表記"] = normalize_text(canon)
