import tempfile
import textwrap

import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import kwja
from kwja.callbacks.char_module_writer import CharModuleWriter
from kwja.datamodule.datasets.char_inference_dataset import CharInferenceDataset
from kwja.utils.constants import WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS


class MockTrainer:
    def __init__(self, predict_dataloaders):
        self.predict_dataloaders = predict_dataloaders


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = CharModuleWriter(tmp_dir)


def test_write_on_batch_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = CharModuleWriter(tmp_dir)
        text = "今日は晴れだぁ"
        tokenizer = AutoTokenizer.from_pretrained("ku-nlp/roberta-base-japanese-char-wwm")
        dataset = CharInferenceDataset(
            texts=ListConfig([text]),
            tokenizer=tokenizer,
            max_seq_length=len(text) + 2,
            document_split_stride=1,
            doc_id_prefix="test",
        )
        trainer = MockTrainer([DataLoader(dataset)])

        word_segmentation_predictions = torch.tensor(
            [
                [
                    WORD_SEGMENTATION_TAGS.index("B"),  # [CLS]
                    WORD_SEGMENTATION_TAGS.index("B"),  # 今
                    WORD_SEGMENTATION_TAGS.index("I"),  # 日
                    WORD_SEGMENTATION_TAGS.index("B"),  # は
                    WORD_SEGMENTATION_TAGS.index("B"),  # 晴
                    WORD_SEGMENTATION_TAGS.index("I"),  # れ
                    WORD_SEGMENTATION_TAGS.index("B"),  # だ
                    WORD_SEGMENTATION_TAGS.index("I"),  # ぁ
                    WORD_SEGMENTATION_TAGS.index("B"),  # [SEP]
                ]
            ],
            dtype=torch.long,
        )
        word_norm_op_predictions = torch.tensor(
            [
                [
                    WORD_NORM_OP_TAGS.index("K"),  # [CLS]
                    WORD_NORM_OP_TAGS.index("K"),  # 今
                    WORD_NORM_OP_TAGS.index("K"),  # 日
                    WORD_NORM_OP_TAGS.index("K"),  # は
                    WORD_NORM_OP_TAGS.index("K"),  # 晴
                    WORD_NORM_OP_TAGS.index("K"),  # れ
                    WORD_NORM_OP_TAGS.index("K"),  # だ
                    WORD_NORM_OP_TAGS.index("D"),  # ぁ
                    WORD_NORM_OP_TAGS.index("K"),  # [SEP]
                ]
            ],
            dtype=torch.long,
        )
        prediction = {
            "example_ids": [0],
            "word_segmentation_predictions": word_segmentation_predictions,
            "word_norm_op_predictions": word_norm_op_predictions,
        }

        writer.write_on_batch_end(trainer, ..., prediction, ..., ..., ..., 0)
        assert writer.destination.read_text() == textwrap.dedent(
            f"""\
            # S-ID:test-0-0 kwja:{kwja.__version__}
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だぁ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
