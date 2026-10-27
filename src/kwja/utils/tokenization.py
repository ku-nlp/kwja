from rhoknp import Morpheme
from tokenizers import Encoding
from transformers import PreTrainedTokenizerBase

from kwja.utils.normalization import normalize_text


def tokenize_morphemes(
    morphemes: list[Morpheme], tokenizer: PreTrainedTokenizerBase, *, add_special_tokens: bool
) -> Encoding:
    """Tokenize normalized surfaces while preserving the original morphemes and their indices."""
    return tokenizer(
        [normalize_text(morpheme.text) for morpheme in morphemes],
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
        is_split_into_words=True,
    ).encodings[0]
