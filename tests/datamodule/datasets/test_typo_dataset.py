from pathlib import Path

from transformers import AutoTokenizer

from kwja.datamodule.datasets.typo_dataset import TypoDataset
from kwja.utils.constants import IGNORE_INDEX

here = Path(__file__).absolute().parent
path = here.joinpath("typo_files")

tokenizer = AutoTokenizer.from_pretrained(
    "ku-nlp/roberta-base-japanese-char-wwm",
    do_word_tokenize=False,
    additional_special_tokens=["<k>", "<d>", "<_>", "<dummy>"],
)


def test_init():
    _ = TypoDataset(str(path), tokenizer)


def test_getitem():
    max_seq_length = 512
    dataset = TypoDataset(str(path), tokenizer, max_seq_length=max_seq_length)
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "example_ids" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "kdr_labels" in item
        assert "ins_labels" in item
        assert item["example_ids"] == i
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["kdr_labels"].shape == (max_seq_length,)
        assert item["ins_labels"].shape == (max_seq_length,)

        kdr_labels = [x for x in item["kdr_labels"] if x != IGNORE_INDEX]
        ins_labels = [x for x in item["ins_labels"].tolist() if x != IGNORE_INDEX]
        assert len(dataset.examples[i].pre_text) == len(kdr_labels) == len(ins_labels) - 1
