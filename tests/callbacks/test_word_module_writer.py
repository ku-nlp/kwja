import tempfile
import textwrap
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig
from rhoknp.props import DepType
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import kwja
from kwja.callbacks.word_module_writer import WordModuleWriter
from kwja.datamodule.datasets.word_inference_dataset import WordInferenceDataset
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAGS,
    DEPENDENCY_TYPES,
    NE_TAGS,
    POS_TAGS,
    SUBPOS_TAGS,
    WORD_FEATURES,
    WordTask,
)
from kwja.utils.jumandic import JumanDic

AMBIG_SURF_SPECS = [
    {
        "conjtype": "イ形容詞アウオ段",
        "conjform": "エ基本形",
    },
    {
        "conjtype": "イ形容詞イ段",
        "conjform": "エ基本形",
    },
    {
        "conjtype": "イ形容詞イ段特殊",
        "conjform": "エ基本形",
    },
]


class MockTrainer:
    def __init__(self, predict_dataloaders):
        self.predict_dataloaders = predict_dataloaders


def make_dummy_jumandic() -> JumanDic:
    with tempfile.TemporaryDirectory() as jumandic_dir:
        JumanDic.build(
            Path(jumandic_dir),
            [
                ["今日", "きょう", "今日", "名詞", "時相名詞", "*", "*", "代表表記:今日/きょう カテゴリ:時間"],
                ["あい", "あい", "あい", "名詞", "普通名詞", "*", "*", "代表表記:愛/あい 漢字読み:音 カテゴリ:抽象物"],
                ["あい", "あい", "あい", "名詞", "普通名詞", "*", "*", "代表表記:藍/あい カテゴリ:植物"],
            ],
        )
        return JumanDic(Path(jumandic_dir))


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = WordModuleWriter(tmp_dir, AMBIG_SURF_SPECS)


def test_write_on_batch_end():
    juman_texts = [
        textwrap.dedent(
            f"""\
            # S-ID:test-0-0 kwja:{kwja.__version__}
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
    ]
    tokens = ["[CLS] 今日 は 晴れ だ"]

    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write("".join(juman_texts))
    juman_file.seek(0)

    reading_logits = torch.zeros(1, 5, 12907, dtype=torch.float)  # 12907: vocab size
    reading_logits[0][1][255] = 1.0  # きょう
    reading_logits[0][2][1] = 1.0  # ID
    reading_logits[0][3][4039] = 1.0  # はれ
    reading_logits[0][4][1] = 1.0  # ID

    reading_subword_map = torch.tensor(
        [
            [
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ],
        dtype=torch.bool,
    )

    pos_logits = torch.zeros(1, 13, len(POS_TAGS), dtype=torch.float)
    pos_logits[0][0][POS_TAGS.index("名詞")] = 1.0
    pos_logits[0][1][POS_TAGS.index("助詞")] = 1.0
    pos_logits[0][2][POS_TAGS.index("名詞")] = 1.0
    pos_logits[0][3][POS_TAGS.index("判定詞")] = 1.0

    subpos_logits = torch.zeros(1, 13, len(SUBPOS_TAGS), dtype=torch.float)
    subpos_logits[0][0][SUBPOS_TAGS.index("時相名詞")] = 1.0
    subpos_logits[0][1][SUBPOS_TAGS.index("副助詞")] = 1.0
    subpos_logits[0][2][SUBPOS_TAGS.index("普通名詞")] = 1.0
    subpos_logits[0][3][SUBPOS_TAGS.index("*")] = 1.0

    conjtype_logits = torch.zeros(1, 13, len(CONJTYPE_TAGS), dtype=torch.float)
    conjtype_logits[0][0][CONJTYPE_TAGS.index("*")] = 1.0
    conjtype_logits[0][1][CONJTYPE_TAGS.index("*")] = 1.0
    conjtype_logits[0][2][CONJTYPE_TAGS.index("*")] = 1.0
    conjtype_logits[0][3][CONJTYPE_TAGS.index("判定詞")] = 1.0

    conjform_logits = torch.zeros(1, 13, len(CONJFORM_TAGS), dtype=torch.float)
    conjform_logits[0][0][CONJFORM_TAGS.index("*")] = 1.0
    conjform_logits[0][1][CONJFORM_TAGS.index("*")] = 1.0
    conjform_logits[0][2][CONJFORM_TAGS.index("*")] = 1.0
    conjform_logits[0][3][CONJFORM_TAGS.index("基本形")] = 1.0

    word_feature_probabilities = torch.zeros(1, 13, len(WORD_FEATURES), dtype=torch.float)
    word_feature_probabilities[0][0][WORD_FEATURES.index("基本句-主辞")] = 1.0
    word_feature_probabilities[0][1][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_feature_probabilities[0][1][WORD_FEATURES.index("文節-区切")] = 1.0
    word_feature_probabilities[0][2][WORD_FEATURES.index("基本句-主辞")] = 1.0
    word_feature_probabilities[0][2][WORD_FEATURES.index("用言表記先頭")] = 1.0
    word_feature_probabilities[0][2][WORD_FEATURES.index("用言表記末尾")] = 1.0
    word_feature_probabilities[0][3][WORD_FEATURES.index("基本句-区切")] = 1.0
    word_feature_probabilities[0][3][WORD_FEATURES.index("文節-区切")] = 1.0

    ne_predictions = torch.full((1, 13), NE_TAGS.index("O"), dtype=torch.long)
    ne_predictions[0][2] = NE_TAGS.index("B-DATE")

    base_phrase_feature_probabilities = torch.zeros(1, 13, len(BASE_PHRASE_FEATURES), dtype=torch.float)
    base_phrase_feature_probabilities[0][0][BASE_PHRASE_FEATURES.index("体言")] = 1.0
    base_phrase_feature_probabilities[0][2][BASE_PHRASE_FEATURES.index("用言:判")] = 1.0
    base_phrase_feature_probabilities[0][2][BASE_PHRASE_FEATURES.index("時制:非過去")] = 1.0
    base_phrase_feature_probabilities[0][2][BASE_PHRASE_FEATURES.index("節-主辞")] = 1.0
    base_phrase_feature_probabilities[0][2][BASE_PHRASE_FEATURES.index("節-区切")] = 1.0
    base_phrase_feature_probabilities[0][2][BASE_PHRASE_FEATURES.index("レベル:C")] = 1.0
    base_phrase_feature_probabilities[0][2][BASE_PHRASE_FEATURES.index("状態述語")] = 1.0

    dependency_topk = 2
    dependency_logits = torch.zeros(1, 4, 13, dtype=torch.float)  # (b, word, word)
    dependency_logits[0][0][2] = 1.0
    dependency_logits[0][1][1] = 1.0
    dependency_logits[0][2][12] = 1.0  # [ROOT]
    dependency_logits[0][3][2] = 1.0

    dependency_type_logits = torch.zeros(1, 13, 1, len(DEPENDENCY_TYPES), dtype=torch.float)
    dependency_type_logits[0][0][0][DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0
    dependency_type_logits[0][1][0][DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0
    dependency_type_logits[0][2][0][DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0
    dependency_type_logits[0][3][0][DEPENDENCY_TYPES.index(DepType.DEPENDENCY)] = 1.0

    cohesion_logits = torch.zeros(1, 6, 4, 13, dtype=torch.float)  # (b, rel, word, word)
    cohesion_logits[0, :, :, 10] = 1.0  # [NULL]
    cohesion_logits[0, :, :, 11] = 1.0  # [NA]
    cohesion_logits[0, 0, 2, 0] = 2.0  # 今日 ガ 晴れ

    # NOTE: This is not a correct prediction, but it is used to test the module.
    discourse_logits = torch.zeros(1, 4, 13, 7, dtype=torch.float)  # (b, word, word, rel)
    discourse_logits[0][2][2][1] = 1.0  # 晴れ -> 晴れ: 原因・理由
    discourse_probabilities = discourse_logits.softmax(dim=3)
    discourse_max_probabilities, discourse_predictions = discourse_probabilities.max(dim=3)

    prediction = {
        "tokens": tokens,
        "example_ids": [0],
        "reading_predictions": reading_logits.argmax(dim=2),
        "reading_subword_map": reading_subword_map,
        "pos_logits": pos_logits,
        "subpos_logits": subpos_logits,
        "conjtype_logits": conjtype_logits,
        "conjform_logits": conjform_logits,
        "word_feature_probabilities": word_feature_probabilities,
        "ne_predictions": ne_predictions,
        "base_phrase_feature_probabilities": base_phrase_feature_probabilities,
        "dependency_predictions": dependency_logits.topk(k=dependency_topk, dim=2).indices,
        "dependency_type_predictions": dependency_type_logits.argmax(dim=3),
        "cohesion_logits": cohesion_logits,
        "discourse_predictions": discourse_predictions,
    }

    output_filename = "test"
    with tempfile.TemporaryDirectory() as tmp_dir:
        # max_seq_length = 6 ([CLS], 今日, は, 晴れ, だ, [SEP]) + 7 (著者, 読者, 不特定:人, 不特定:物, [NULL], [NA], [ROOT])
        writer = WordModuleWriter(tmp_dir, AMBIG_SURF_SPECS, output_filename=output_filename)
        writer.jumandic = make_dummy_jumandic()

        exophora_referents = ["著者", "読者", "不特定:人", "不特定:物"]
        special_tokens = exophora_referents + ["[NULL]", "[NA]", "[ROOT]"]
        tokenizer = AutoTokenizer.from_pretrained(
            "nlp-waseda/roberta-base-japanese", additional_special_tokens=special_tokens
        )
        dataset = WordInferenceDataset(
            tokenizer=tokenizer,
            document_split_stride=1,
            cohesion_tasks=ListConfig(["pas_analysis", "bridging_reference_resolution", "coreference_resolution"]),
            exophora_referents=ListConfig(exophora_referents),
            restrict_cohesion_target=True,
            pas_cases=ListConfig(["ガ", "ヲ", "ニ", "ガ２"]),
            br_cases=ListConfig(["ノ"]),
            special_tokens=ListConfig(special_tokens),
            juman_file=Path(juman_file.name),
            max_seq_length=13,
        )

        trainer = MockTrainer([DataLoader(dataset)])
        module = pl.LightningModule()
        module.training_tasks = [
            WordTask.READING_PREDICTION,
            WordTask.MORPHOLOGICAL_ANALYSIS,
            WordTask.WORD_FEATURE_TAGGING,
            WordTask.NER,
            WordTask.BASE_PHRASE_FEATURE_TAGGING,
            WordTask.DEPENDENCY_PARSING,
            WordTask.COHESION_ANALYSIS,
            WordTask.DISCOURSE_PARSING,
        ]

        writer.write_on_batch_end(trainer, module, prediction, ..., ..., 0, 0)  # noqa
        expected_knp = textwrap.dedent(
            f"""\
            # S-ID:test-0-0 kwja:{kwja.__version__}
            * 1D
            + 1D <体言>
            今日 きょう 今日 名詞 6 時相名詞 10 * 0 * 0 "代表表記:今日/きょう カテゴリ:時間" <基本句-主辞>
            は は は 助詞 9 副助詞 2 * 0 * 0
            * -1D
            + -1D <rel type="ガ" target="今日" sid="test-0-0" id="0"/><NE:DATE:晴れ><用言:判><時制:非過去><レベル:C><状態述語><節-区切><節-主辞><談話関係:test-0-0/1/原因・理由>
            晴れ はれ 晴れ 名詞 6 普通名詞 1 * 0 * 0 <基本句-主辞><用言表記先頭><用言表記末尾>
            だ だ だ 判定詞 4 * 0 判定詞 25 基本形 2
            EOS
            """
        )
        assert Path(tmp_dir).joinpath(f"{output_filename}.knp").read_text() == expected_knp
        juman_file.close()
