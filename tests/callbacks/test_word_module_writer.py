import tempfile
import textwrap
from pathlib import Path

import torch
from rhoknp.props import DepType

import jula
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
    texts = ["今日 は 晴れ だ"]

    word_analysis_pos_logits = torch.zeros(1, 11, len(POS_TYPES), dtype=torch.float)
    word_analysis_pos_logits[0][0][POS_TYPES.index("名詞")] = 1.0
    word_analysis_pos_logits[0][1][POS_TYPES.index("助詞")] = 1.0
    word_analysis_pos_logits[0][2][POS_TYPES.index("名詞")] = 1.0
    word_analysis_pos_logits[0][3][POS_TYPES.index("判定詞")] = 1.0

    word_analysis_subpos_logits = torch.zeros(1, 11, len(SUBPOS_TYPES), dtype=torch.float)
    word_analysis_subpos_logits[0][0][SUBPOS_TYPES.index("時相名詞")] = 1.0
    word_analysis_subpos_logits[0][1][SUBPOS_TYPES.index("副助詞")] = 1.0
    word_analysis_subpos_logits[0][2][SUBPOS_TYPES.index("普通名詞")] = 1.0
    word_analysis_subpos_logits[0][3][SUBPOS_TYPES.index("*")] = 1.0

    word_analysis_conjtype_logits = torch.zeros(1, 11, len(CONJTYPE_TYPES), dtype=torch.float)
    word_analysis_conjtype_logits[0][0][CONJTYPE_TYPES.index("*")] = 1.0
    word_analysis_conjtype_logits[0][1][CONJTYPE_TYPES.index("*")] = 1.0
    word_analysis_conjtype_logits[0][2][CONJTYPE_TYPES.index("*")] = 1.0
    word_analysis_conjtype_logits[0][3][CONJTYPE_TYPES.index("判定詞")] = 1.0

    word_analysis_conjform_logits = torch.zeros(1, 11, len(CONJFORM_TYPES), dtype=torch.float)
    word_analysis_conjform_logits[0][0][CONJFORM_TYPES.index("*")] = 1.0
    word_analysis_conjform_logits[0][1][CONJFORM_TYPES.index("*")] = 1.0
    word_analysis_conjform_logits[0][2][CONJFORM_TYPES.index("*")] = 1.0
    word_analysis_conjform_logits[0][3][CONJFORM_TYPES.index("基本形")] = 1.0

    word_feature_logits = torch.zeros(1, 11, len(WORD_FEATURES), dtype=torch.float)
    word_feature_logits[0][0][WORD_FEATURES.index("基本句-主辞")] = 1.0
    word_feature_logits[0][1][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_feature_logits[0][1][WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_logits[0][2][WORD_FEATURES.index("基本句-主辞")] = 1.0
    word_feature_logits[0][2][WORD_FEATURES.index("用言表記先頭")] = 1.0
    word_feature_logits[0][2][WORD_FEATURES.index("用言表記末尾")] = 1.0
    word_feature_logits[0][3][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_feature_logits[0][3][WORD_FEATURES.index("文節-区切")] = 1.0

    base_phrase_feature_logits = torch.zeros(1, 11, len(BASE_PHRASE_FEATURES), dtype=torch.float)
    base_phrase_feature_logits[0][0][BASE_PHRASE_FEATURES.index("体言")] = 1.0
    base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("用言:判")] = 1.0
    base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("レベル:C")] = 1.0
    base_phrase_feature_logits[0][2][BASE_PHRASE_FEATURES.index("状態述語")] = 1.0

    dependency_logits = torch.zeros(1, 4, 11, dtype=torch.float)  # (b, word, word)
    dependency_logits[0][0][2] = 1.0
    dependency_logits[0][1][1] = 1.0
    dependency_logits[0][2][10] = 1.0  # [ROOT]
    dependency_logits[0][3][2] = 1.0

    dependency_type_logits = torch.zeros(1, 11, 1, len(DEPENDENCY_TYPE2INDEX), dtype=torch.float)
    dependency_type_logits[0][0][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0
    dependency_type_logits[0][1][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0
    dependency_type_logits[0][2][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0
    dependency_type_logits[0][3][0][DEPENDENCY_TYPE2INDEX[DepType.DEPENDENCY]] = 1.0

    cohesion_logits = torch.zeros(1, 6, 4, 11, dtype=torch.float)  # (b, rel, word, word)
    cohesion_logits[0, :, :, 8] = 1.0  # [NULL]
    cohesion_logits[0, :, :, 9] = 1.0  # [NA]
    cohesion_logits[0, 0, 2, 0] = 2.0  # 今日 ガ 晴れ

    # NOTE: This is not a correct prediction, but it is used to test the module.
    discourse_parsing_logits = torch.zeros(1, 4, 11, 7, dtype=torch.float)  # (b, word, word, rel)
    discourse_parsing_logits[0][2][2][1] = 1.0  # 晴れ -> 晴れ: 原因・理由

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
                "cohesion_logits": cohesion_logits,
                "discourse_parsing_logits": discourse_parsing_logits,
            }
        ]
    ]

    pred_filename = "test"
    with tempfile.TemporaryDirectory() as tmp_dir:
        # max_seq_length = 4 (今日, は, 晴れ, だ) + 7 (著者, 読者, 不特定:人, 不特定:物, [NULL], [NA], [ROOT])
        writer = WordModuleWriter(tmp_dir, pred_filename=pred_filename, max_seq_length=11)
        writer.write_on_epoch_end(..., ..., predictions)
        expected_knp = textwrap.dedent(
            f"""\
            # S-ID:1 jula:{jula.__version__}
            * 1D
            + 1D <体言>
            今日 今日 今日 名詞 0 時相名詞 0 * 0 * 0 <基本句-主辞>
            は は は 助詞 0 副助詞 0 * 0 * 0
            * -1D
            + -1D <rel type="ガ" target="今日" sid="1" id="0"/><用言:判><時制:非過去><節-主辞><節-区切><レベル:C><状態述語><談話関係:1/1/原因・理由>
            晴れ 晴れ 晴れ 名詞 0 普通名詞 0 * 0 * 0 <基本句-主辞><用言表記先頭><用言表記末尾>
            だ だ だ 判定詞 0 * 0 判定詞 0 基本形 0
            EOS
            """
        )
        assert Path(tmp_dir).joinpath(f"{pred_filename}.knp").read_text() == expected_knp
