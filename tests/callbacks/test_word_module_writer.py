import tempfile
import textwrap

import torch
from rhoknp.units.utils import DepType

from jula.callbacks.word_module_writer import WordModuleWriter
from jula.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TYPES,
    CONJTYPE_TYPES,
    DEPENDENCY_TYPE2INDEX,
    POS_TYPES,
    SUBPOS_TYPES,
    WORD_FEATURES,
)


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = WordModuleWriter(
            tmp_dir,
            tokenizer_kwargs={
                "additional_special_tokens": ["著者", "読者", "不特定:人", "不特定:物", "[NULL]", "[NA]", "[ROOT]"],
            },
        )


def test_write_on_epoch_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = WordModuleWriter(tmp_dir)

        texts = ["今日 は 晴れ だ"]

        word_analysis_pos_logits = torch.zeros(1, 7, len(POS_TYPES), dtype=torch.float)
        word_analysis_pos_logits[0][0][POS_TYPES.index("名詞")] = 1.0
        word_analysis_pos_logits[0][1][POS_TYPES.index("助詞")] = 1.0
        word_analysis_pos_logits[0][2][POS_TYPES.index("名詞")] = 1.0
        word_analysis_pos_logits[0][3][POS_TYPES.index("判定詞")] = 1.0

        word_analysis_subpos_logits = torch.zeros(1, 7, len(SUBPOS_TYPES), dtype=torch.float)
        word_analysis_subpos_logits[0][0][SUBPOS_TYPES.index("時相名詞")] = 1.0
        word_analysis_subpos_logits[0][1][SUBPOS_TYPES.index("副助詞")] = 1.0
        word_analysis_subpos_logits[0][2][SUBPOS_TYPES.index("普通名詞")] = 1.0
        word_analysis_subpos_logits[0][3][SUBPOS_TYPES.index("*")] = 1.0

        word_analysis_conjtype_logits = torch.zeros(1, 7, len(CONJTYPE_TYPES), dtype=torch.float)
        word_analysis_conjtype_logits[0][0][CONJTYPE_TYPES.index("*")] = 1.0
        word_analysis_conjtype_logits[0][1][CONJTYPE_TYPES.index("*")] = 1.0
        word_analysis_conjtype_logits[0][2][CONJTYPE_TYPES.index("*")] = 1.0
        word_analysis_conjtype_logits[0][3][CONJTYPE_TYPES.index("判定詞")] = 1.0

        word_analysis_conjform_logits = torch.zeros(1, 7, len(CONJFORM_TYPES), dtype=torch.float)
        word_analysis_conjform_logits[0][0][CONJFORM_TYPES.index("*")] = 1.0
        word_analysis_conjform_logits[0][1][CONJFORM_TYPES.index("*")] = 1.0
        word_analysis_conjform_logits[0][2][CONJFORM_TYPES.index("*")] = 1.0
        word_analysis_conjform_logits[0][3][CONJFORM_TYPES.index("基本形")] = 1.0

        word_feature_logits = torch.zeros(1, 7, len(WORD_FEATURES), dtype=torch.float)
        word_feature_logits[0][0][WORD_FEATURES.index("基本句-主辞")] = 1.0
        word_feature_logits[0][1][WORD_FEATURES.index("基本句-区切")] = 1.0
        word_feature_logits[0][1][WORD_FEATURES.index("文節-区切")] = 1.0
        word_feature_logits[0][2][WORD_FEATURES.index("基本句-主辞")] = 1.0
        word_feature_logits[0][3][WORD_FEATURES.index("基本句-区切")] = 1.0
        word_feature_logits[0][3][WORD_FEATURES.index("文節-区切")] = 1.0

        base_phrase_feature_logits = torch.zeros(1, 7, len(BASE_PHRASE_FEATURES), dtype=torch.float)
        base_phrase_feature_logits[0][0][BASE_PHRASE_FEATURES.index("体言")] = 1.0
        base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("用言:判")] = 1.0
        base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
        base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
        base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("節-区切")] = 1.0

        dependency_logits = torch.zeros(1, 4, 7, dtype=torch.float)
        dependency_logits[0][0][2] = 1.0
        dependency_logits[0][1][1] = 1.0
        dependency_logits[0][2][6] = 1.0
        dependency_logits[0][3][2] = 1.0

        dependency_type_logits = torch.zeros(1, 7, 1, len(DEPENDENCY_TYPE2INDEX), dtype=torch.float)
        dependency_type_logits[0][0][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0
        dependency_type_logits[0][1][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0
        dependency_type_logits[0][2][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0
        dependency_type_logits[0][3][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0

        predictions = [
            [
                {
                    "texts": texts,
                    "word_analysis_pos_logits": word_analysis_pos_logits,
                    "word_analysis_subpos_logits": word_analysis_subpos_logits,
                    "word_analysis_conjtype_logits": word_analysis_conjtype_logits,
                    "word_analysis_conjform_logits": word_analysis_conjform_logits,
                    "word_feature_logits": word_feature_logits,
                    "base_phrase_feature_logits": base_phrase_feature_logits,
                    "dependency_logits": dependency_logits,
                    "dependency_type_logits": dependency_type_logits,
                }
            ]
        ]
        writer.write_on_epoch_end(..., ..., predictions)
        with open(writer.output_path) as f:
            assert (
                f.read().strip()
                == textwrap.dedent(
                    """\
                *
                + 1D <体言>
                今日 名詞 時相名詞 * * <基本句-主辞>
                は 助詞 副助詞 * *
                *
                + -1D <用言:判><時制:非過去><節-主辞><節-区切>
                晴れ 名詞 普通名詞 * * <基本句-主辞>
                だ 判定詞 * 判定詞 基本形
                EOS
                """
                ).strip()
            )
