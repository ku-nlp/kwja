from pathlib import Path

from jula.datamodule.datasets.char_dataset import CharDataset
from jula.utils.utils import IGNORE_INDEX

here = Path(__file__).absolute().parent
path = here.joinpath("knp_files")


def test_init():
    _ = CharDataset(str(path))


def test_getitem():
    max_seq_length = 512
    # TODO: use roberta
    dataset = CharDataset(
        str(path),
        model_name_or_path="cl-tohoku/bert-base-japanese-char",
        max_seq_length=max_seq_length,
        tokenizer_kwargs={"do_word_tokenize": False},
    )
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "seg_labels" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["seg_labels"].shape == (max_seq_length,)

        cls_token_position = (
            item["input_ids"].tolist().index(dataset.tokenizer.cls_token_id)
        )
        sep_token_position = (
            item["input_ids"].tolist().index(dataset.tokenizer.sep_token_id)
        )
        assert item["seg_labels"][cls_token_position].tolist() == IGNORE_INDEX
        assert item["seg_labels"][sep_token_position].tolist() == IGNORE_INDEX
