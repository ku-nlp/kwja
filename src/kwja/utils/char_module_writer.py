from typing import List, Set, Tuple

from rhoknp import Document, Morpheme

from kwja.utils.constants import WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS
from kwja.utils.word_normalization import get_normalized


def convert_predictions_into_tags(
    word_segmentation_predictions: List[int],
    word_norm_op_predictions: List[int],
    input_ids: List[int],
    special_ids: Set[int],
) -> Tuple[List[str], List[str]]:
    indices = [i for i, input_id in enumerate(input_ids) if input_id not in special_ids]
    word_segmentation_tags = [WORD_SEGMENTATION_TAGS[word_segmentation_predictions[i]] for i in indices]
    word_norm_op_tags = [WORD_NORM_OP_TAGS[word_norm_op_predictions[i]] for i in indices]
    return word_segmentation_tags, word_norm_op_tags


def set_morphemes(document: Document, word_segmentation_tags: List[str], word_norm_op_tags: List[str]) -> None:
    char_index = 0
    for sentence in document.sentences:
        Morpheme.count = 0
        morphemes: List[Morpheme] = []
        surf: str = ""
        ops: List[str] = []
        for char in sentence.text:
            if word_segmentation_tags[char_index] == "B" and surf:
                norm = get_normalized(surf, ops, strict=False)
                morphemes.append(_build_morpheme(surf, norm))
                surf = ""
                ops = []
            surf += char
            ops.append(word_norm_op_tags[char_index])
            char_index += 1
        if surf:
            norm = get_normalized(surf, ops, strict=False)
            morphemes.append(_build_morpheme(surf, norm))
        sentence.morphemes = morphemes


def _build_morpheme(surf: str, norm: str) -> Morpheme:
    return Morpheme(
        surf,
        reading="_",
        lemma=norm or surf,
        pos="未定義語",
        pos_id=15,
        subpos="その他",
        subpos_id=1,
        conjtype="*",
        conjtype_id=0,
        conjform="*",
        conjform_id=0,
    )
