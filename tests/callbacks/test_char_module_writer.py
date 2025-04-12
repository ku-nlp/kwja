from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Optional, Union

import pytest
import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from kwja.callbacks import CharModuleWriter
from kwja.datamodule.datasets import CharInferenceDataset
from kwja.utils.constants import SENT_SEGMENTATION_TAGS, WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS


class MockTrainer:
    def __init__(self, predict_dataloaders: list[DataLoader]):
        self.predict_dataloaders = predict_dataloaders


@pytest.mark.parametrize(
    "destination",
    [
        None,
        Path(TemporaryDirectory().name) / Path("char_prediction.juman"),
        str(Path(TemporaryDirectory().name) / Path("char_prediction.juman")),
    ],
)
def test_init(destination: Optional[Union[str, Path]]):
    _ = CharModuleWriter(destination=destination)


def test_write_on_batch_end(char_tokenizer: PreTrainedTokenizerBase):
    texts = ListConfig(["花咲ガニを買ぅ", "うまそーですね〜〜"])
    max_seq_length = 32
    doc_id_prefix = "test"
    dataset = CharInferenceDataset(
        texts,
        char_tokenizer,
        max_seq_length,
        doc_id_prefix=doc_id_prefix,
    )
    num_examples = len(dataset)

    trainer = MockTrainer([DataLoader(dataset, batch_size=num_examples)])

    sent_segmentation_predictions = torch.zeros((num_examples, max_seq_length), dtype=torch.long)
    # [:, 0] = [CLS]
    sent_segmentation_predictions[0, 1] = SENT_SEGMENTATION_TAGS.index("B")  # 花
    sent_segmentation_predictions[0, 2] = SENT_SEGMENTATION_TAGS.index("I")  # 咲
    sent_segmentation_predictions[0, 3] = SENT_SEGMENTATION_TAGS.index("I")  # ガ
    sent_segmentation_predictions[0, 4] = SENT_SEGMENTATION_TAGS.index("I")  # ニ
    sent_segmentation_predictions[0, 5] = SENT_SEGMENTATION_TAGS.index("I")  # を
    sent_segmentation_predictions[0, 6] = SENT_SEGMENTATION_TAGS.index("I")  # 買
    sent_segmentation_predictions[0, 7] = SENT_SEGMENTATION_TAGS.index("I")  # ぅ
    sent_segmentation_predictions[1, 1] = SENT_SEGMENTATION_TAGS.index("B")  # う
    sent_segmentation_predictions[1, 2] = SENT_SEGMENTATION_TAGS.index("I")  # ま
    sent_segmentation_predictions[1, 3] = SENT_SEGMENTATION_TAGS.index("I")  # そ
    sent_segmentation_predictions[1, 4] = SENT_SEGMENTATION_TAGS.index("I")  # ー
    sent_segmentation_predictions[1, 5] = SENT_SEGMENTATION_TAGS.index("I")  # で
    sent_segmentation_predictions[1, 6] = SENT_SEGMENTATION_TAGS.index("I")  # す
    sent_segmentation_predictions[1, 7] = SENT_SEGMENTATION_TAGS.index("I")  # ね
    sent_segmentation_predictions[1, 8] = SENT_SEGMENTATION_TAGS.index("I")  # 〜
    sent_segmentation_predictions[1, 9] = SENT_SEGMENTATION_TAGS.index("I")  # 〜

    word_segmentation_predictions = torch.zeros((num_examples, max_seq_length), dtype=torch.long)
    # [:, 0] = [CLS]
    word_segmentation_predictions[0, 1] = WORD_SEGMENTATION_TAGS.index("B")  # 花
    word_segmentation_predictions[0, 2] = WORD_SEGMENTATION_TAGS.index("I")  # 咲
    word_segmentation_predictions[0, 3] = WORD_SEGMENTATION_TAGS.index("B")  # ガ
    word_segmentation_predictions[0, 4] = WORD_SEGMENTATION_TAGS.index("I")  # ニ
    word_segmentation_predictions[0, 5] = WORD_SEGMENTATION_TAGS.index("B")  # を
    word_segmentation_predictions[0, 6] = WORD_SEGMENTATION_TAGS.index("B")  # 買
    word_segmentation_predictions[0, 7] = WORD_SEGMENTATION_TAGS.index("I")  # ぅ
    word_segmentation_predictions[1, 1] = WORD_SEGMENTATION_TAGS.index("B")  # う
    word_segmentation_predictions[1, 2] = WORD_SEGMENTATION_TAGS.index("I")  # ま
    word_segmentation_predictions[1, 3] = WORD_SEGMENTATION_TAGS.index("B")  # そ
    word_segmentation_predictions[1, 4] = WORD_SEGMENTATION_TAGS.index("I")  # ー
    word_segmentation_predictions[1, 5] = WORD_SEGMENTATION_TAGS.index("I")  # で
    word_segmentation_predictions[1, 6] = WORD_SEGMENTATION_TAGS.index("I")  # す
    word_segmentation_predictions[1, 7] = WORD_SEGMENTATION_TAGS.index("B")  # ね
    word_segmentation_predictions[1, 8] = WORD_SEGMENTATION_TAGS.index("I")  # 〜
    word_segmentation_predictions[1, 9] = WORD_SEGMENTATION_TAGS.index("I")  # 〜

    word_norm_op_predictions = torch.zeros((num_examples, max_seq_length), dtype=torch.long)
    # [:, 0] = [CLS]
    word_norm_op_predictions[0, 1] = WORD_NORM_OP_TAGS.index("K")  # 花
    word_norm_op_predictions[0, 2] = WORD_NORM_OP_TAGS.index("K")  # 咲
    word_norm_op_predictions[0, 3] = WORD_NORM_OP_TAGS.index("V")  # ガ
    word_norm_op_predictions[0, 4] = WORD_NORM_OP_TAGS.index("K")  # ニ
    word_norm_op_predictions[0, 5] = WORD_NORM_OP_TAGS.index("K")  # を
    word_norm_op_predictions[0, 6] = WORD_NORM_OP_TAGS.index("K")  # 買
    word_norm_op_predictions[0, 7] = WORD_NORM_OP_TAGS.index("S")  # ぅ
    word_norm_op_predictions[1, 1] = WORD_NORM_OP_TAGS.index("K")  # う
    word_norm_op_predictions[1, 2] = WORD_NORM_OP_TAGS.index("K")  # ま
    word_norm_op_predictions[1, 3] = WORD_NORM_OP_TAGS.index("K")  # そ
    word_norm_op_predictions[1, 4] = WORD_NORM_OP_TAGS.index("P")  # ー
    word_norm_op_predictions[1, 5] = WORD_NORM_OP_TAGS.index("K")  # で
    word_norm_op_predictions[1, 6] = WORD_NORM_OP_TAGS.index("K")  # す
    word_norm_op_predictions[1, 7] = WORD_NORM_OP_TAGS.index("K")  # ね
    word_norm_op_predictions[1, 8] = WORD_NORM_OP_TAGS.index("E")  # 〜
    word_norm_op_predictions[1, 9] = WORD_NORM_OP_TAGS.index("D")  # 〜

    prediction = {
        "example_ids": torch.arange(num_examples, dtype=torch.long),
        "sent_segmentation_predictions": sent_segmentation_predictions,
        "word_segmentation_predictions": word_segmentation_predictions,
        "word_norm_op_predictions": word_norm_op_predictions,
    }

    with TemporaryDirectory() as tmp_dir:
        writer = CharModuleWriter(destination=tmp_dir / Path("char_prediction.juman"))
        writer.write_on_batch_end(trainer, ..., prediction, None, ..., 0, 0)  # type: ignore
        assert isinstance(writer.destination, Path), "destination isn't set"
        assert writer.destination.read_text() == dedent(
            f"""\
            # S-ID:{doc_id_prefix}-0-1 kwja:{version("kwja")}
            花咲 _ 花咲 未定義語 15 その他 1 * 0 * 0
            ガニ _ カニ 未定義語 15 その他 1 * 0 * 0
            を _ を 未定義語 15 その他 1 * 0 * 0
            買ぅ _ 買う 未定義語 15 その他 1 * 0 * 0
            EOS
            # S-ID:{doc_id_prefix}-1-1 kwja:{version("kwja")}
            うま _ うま 未定義語 15 その他 1 * 0 * 0
            そーです _ そうです 未定義語 15 その他 1 * 0 * 0
            ね〜〜 _ ねえ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
