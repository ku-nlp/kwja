import tempfile
import textwrap
from pathlib import Path

import torch
from omegaconf import ListConfig
from rhoknp.props import DepType
from torch.utils.data import DataLoader

import jula
from jula.callbacks.word_module_writer import WordModuleWriter
from jula.datamodule.datasets.word_inference_dataset import WordInferenceDataset
from jula.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TYPES,
    CONJTYPE_TYPES,
    DEPENDENCY_TYPE2INDEX,
    NE_TAGS,
    POS_TYPES,
    SUBPOS_TYPES,
    WORD_FEATURES,
)

here = Path(__file__).absolute().parent
reading_resource_path = here.parent / "datamodule/datasets/reading_files"


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = WordModuleWriter(tmp_dir, str(reading_resource_path))


class MockTrainer:
    def __init__(self, predict_dataloaders):
        self.predict_dataloaders = predict_dataloaders


def test_write_on_epoch_end():
    texts = ["今日 は 晴れ だ"]
    tokens = ["[CLS] 今日 は 晴れ だ"]

    reading_subword_map = torch.tensor(
        [
            [
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
                [False, True, False, False, False],
            ]
        ],
        dtype=torch.bool,
    )
    reading_prediction_logits = torch.zeros(1, 5, 12906 + 2, dtype=torch.float)
    reading_prediction_logits[0][1][257] = 1.0  # きょう
    reading_prediction_logits[0][1][1] = 1.0  # ID
    reading_prediction_logits[0][1][4041] = 1.0  # はれ
    reading_prediction_logits[0][1][1] = 1.0  # ID

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

    ne_logits = torch.zeros(1, 11, len(NE_TAGS), dtype=torch.float)
    ne_logits[0][0][NE_TAGS.index("O")] = 1.0
    ne_logits[0][1][NE_TAGS.index("O")] = 1.0
    ne_logits[0][2][NE_TAGS.index("B-DATE")] = 1.0
    ne_logits[0][3][NE_TAGS.index("O")] = 1.0

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
                "tokens": tokens,
                "dataloader_idx": 0,
                "reading_subword_map": reading_subword_map,
                "reading_prediction_logits": reading_prediction_logits,
                "word_analysis_pos_logits": word_analysis_pos_logits,
                "word_analysis_subpos_logits": word_analysis_subpos_logits,
                "word_analysis_conjtype_logits": word_analysis_conjtype_logits,
                "word_analysis_conjform_logits": word_analysis_conjform_logits,
                "ne_logits": ne_logits,
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
        writer = WordModuleWriter(tmp_dir, str(reading_resource_path), pred_filename=pred_filename)
        exophora_referents = ["著者", "読者", "不特定:人", "不特定:物"]
        special_tokens = exophora_referents + ["[NULL]", "[NA]", "[ROOT]"]
        dataset = WordInferenceDataset(
            texts=texts,
            model_name_or_path="nlp-waseda/roberta-base-japanese",
            max_seq_length=11,
            tokenizer_kwargs={"additional_special_tokens": special_tokens},
            pas_cases=ListConfig(["ガ", "ヲ", "ニ", "ガ２"]),
            bar_rels=ListConfig(["ノ"]),
            exophora_referents=ListConfig(exophora_referents),
            cohesion_tasks=ListConfig(["pas_analysis", "bridging", "coreference"]),
            special_tokens=ListConfig(special_tokens),
        )
        trainer = MockTrainer([DataLoader(dataset)])
        writer.write_on_epoch_end(trainer, ..., predictions)
        expected_knp = textwrap.dedent(
            f"""\
            # S-ID:1 jula:{jula.__version__}
            * 1D
            + 1D <体言>
            今日 今日 今日 名詞 6 時相名詞 10 * 0 * 0 <基本句-主辞>
            は は は 助詞 9 副助詞 2 * 0 * 0
            * -1D
            + -1D <rel type="ガ" target="今日" sid="1" id="0"/><NE:DATE:晴れ><用言:判><時制:非過去><節-主辞><節-区切><レベル:C><状態述語><談話関係:1/1/原因・理由>
            晴れ 晴れ 晴れ 名詞 6 普通名詞 1 * 0 * 0 <基本句-主辞><用言表記先頭><用言表記末尾>
            だ だ だ 判定詞 4 * 0 判定詞 25 基本形 2
            EOS
            """
        )
        assert Path(tmp_dir).joinpath(f"{pred_filename}.knp").read_text() == expected_knp
