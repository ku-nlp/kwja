from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import List, Optional, Union

import pytest
import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

import kwja
from kwja.callbacks.senter_module_writer import SenterModuleWriter
from kwja.callbacks.utils import SENT_SEGMENTATION_TAGS
from kwja.datamodule.datasets import SenterInferenceDataset


class MockTrainer:
    def __init__(self, predict_dataloaders: List[DataLoader]):
        self.predict_dataloaders = predict_dataloaders


@pytest.mark.parametrize(
    "destination",
    [
        None,
        Path(TemporaryDirectory().name) / Path("senter_prediction.txt"),
        str(Path(TemporaryDirectory().name) / Path("senter_prediction.txt")),
    ],
)
def test_init(destination: Optional[Union[str, Path]]):
    _ = SenterModuleWriter(destination=destination)


def test_write_on_batch_end(char_tokenizer: PreTrainedTokenizerBase) -> None:
    texts = ["一文目。二文目。", "違う文書の一文目。二文目。"]
    num_examples = 2
    max_seq_length = 32

    probabilities = torch.zeros((num_examples, max_seq_length, len(SENT_SEGMENTATION_TAGS)), dtype=torch.float)
    probabilities[0, 0, SENT_SEGMENTATION_TAGS.index("B")] = 1.0  # [CLS]
    probabilities[0, 1, SENT_SEGMENTATION_TAGS.index("B")] = 1.0  # 一
    probabilities[0, 2, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 文
    probabilities[0, 3, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 目
    probabilities[0, 4, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 。
    probabilities[0, 5, SENT_SEGMENTATION_TAGS.index("B")] = 1.0  # 二
    probabilities[0, 6, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 文
    probabilities[0, 7, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 目
    probabilities[0, 8, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 。
    probabilities[0, 9, SENT_SEGMENTATION_TAGS.index("B")] = 1.0  # [SEP]
    probabilities[1, 0, SENT_SEGMENTATION_TAGS.index("B")] = 1.0  # [CLS]
    probabilities[1, 1, SENT_SEGMENTATION_TAGS.index("B")] = 1.0  # 違
    probabilities[1, 2, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # う
    probabilities[1, 3, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 文
    probabilities[1, 4, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 書
    probabilities[1, 5, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # の
    probabilities[1, 6, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 一
    probabilities[1, 7, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 文
    probabilities[1, 8, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 目
    probabilities[1, 9, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 。
    probabilities[1, 10, SENT_SEGMENTATION_TAGS.index("B")] = 1.0  # 二
    probabilities[1, 11, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 文
    probabilities[1, 12, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 目
    probabilities[1, 13, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # 。
    probabilities[1, 14, SENT_SEGMENTATION_TAGS.index("I")] = 1.0  # [SEP]

    predictions = probabilities.argmax(dim=2)
    prediction = {
        "example_ids": torch.arange(num_examples, dtype=torch.long),
        "sent_segmentation_predictions": predictions,
    }

    doc_id_prefix = "test"
    with TemporaryDirectory() as tmp_dir:
        destination = tmp_dir / Path("senter_prediction.txt")
        expected_texts = [
            dedent(
                f"""\
                # S-ID:{doc_id_prefix}-0-1 kwja:{kwja.__version__}
                一文目。
                # S-ID:{doc_id_prefix}-0-2 kwja:{kwja.__version__}
                二文目。
                # S-ID:{doc_id_prefix}-1-1 kwja:{kwja.__version__}
                違う文書の一文目。
                # S-ID:{doc_id_prefix}-1-2 kwja:{kwja.__version__}
                二文目。
                """
            ),
        ]
        dataset = SenterInferenceDataset(ListConfig(texts), char_tokenizer, max_seq_length, doc_id_prefix=doc_id_prefix)
        trainer = MockTrainer([DataLoader(dataset, batch_size=num_examples)])
        writer = SenterModuleWriter(destination=destination)
        writer.write_on_batch_end(trainer, ..., prediction, None, ..., 0, 0)
        writer.on_predict_epoch_end(trainer, ...)
        assert isinstance(writer.destination, Path), "destination isn't set"
        assert writer.destination.read_text() == expected_texts[0]
