from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import List, Optional, Union

import pytest
import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import kwja
from kwja.callbacks.char_module_writer import CharModuleWriter
from kwja.datamodule.datasets import CharInferenceDataset
from kwja.utils.constants import WORD_NORM_OP_TAGS, WORD_SEGMENTATION_TAGS


class MockTrainer:
    def __init__(self, predict_dataloaders: List[DataLoader]):
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


def test_write_on_batch_end():
    texts = ["花咲ガニを買ぅ", "うまそーですね〜〜"]
    tokenizer = AutoTokenizer.from_pretrained("ku-nlp/roberta-base-japanese-char-wwm", do_word_tokenize=False)
    max_seq_length = 20
    doc_id_prefix = "test"
    dataset = CharInferenceDataset(
        texts=ListConfig(texts),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        document_split_stride=-1,
        doc_id_prefix=doc_id_prefix,
    )
    num_examples = len(dataset)

    trainer = MockTrainer([DataLoader(dataset, batch_size=num_examples)])

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
        "word_segmentation_predictions": word_segmentation_predictions,
        "word_norm_op_predictions": word_norm_op_predictions,
    }

    with TemporaryDirectory() as tmp_dir:
        writer = CharModuleWriter(destination=tmp_dir / Path("char_prediction.juman"))
        writer.write_on_batch_end(trainer, ..., prediction, None, ..., ..., 0)
        assert isinstance(writer.destination, Path), "destination isn't set"
        assert writer.destination.read_text() == dedent(
            f"""\
            # S-ID:{doc_id_prefix}-0-0 kwja:{kwja.__version__}
            花咲 _ 花咲 未定義語 15 その他 1 * 0 * 0
            ガニ _ カニ 未定義語 15 その他 1 * 0 * 0
            を _ を 未定義語 15 その他 1 * 0 * 0
            買ぅ _ 買う 未定義語 15 その他 1 * 0 * 0
            EOS
            # S-ID:{doc_id_prefix}-1-0 kwja:{kwja.__version__}
            うま _ うま 未定義語 15 その他 1 * 0 * 0
            そーです _ そうです 未定義語 15 その他 1 * 0 * 0
            ね〜〜 _ ねえ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
