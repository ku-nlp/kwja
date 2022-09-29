from pathlib import Path

from kwja.datamodule.datasets.typo_dataset import TypoDataset

here = Path(__file__).absolute().parent
path = here.joinpath("typo_files")


def test_init():
    _ = TypoDataset(path=str(path), extended_vocab_path=f"{path}/extended_vocab.txt")


def test_getitem():
    max_seq_length = 512
    # TODO: use roberta
    dataset = TypoDataset(
        path=str(path),
        extended_vocab_path=f"{path}/extended_vocab.txt",
        model_name_or_path="ku-nlp/roberta-base-japanese-char-wwm",
        max_seq_length=max_seq_length,
        tokenizer_kwargs={
            "do_word_tokenize": False,
            "additional_special_tokens": ["<k>", "<d>", "<_>", "<dummy>"],
        },
    )
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "kdr_labels" in item
        assert "ins_labels" in item
        assert item["input_ids"].shape == (max_seq_length,)
        assert item["attention_mask"].shape == (max_seq_length,)
        assert item["kdr_labels"].shape == (max_seq_length,)
        assert item["ins_labels"].shape == (max_seq_length,)

        kdr_labels = [x for x in item["kdr_labels"] if x != dataset.tokenizer.pad_token_id]
        ins_labels = [x for x in item["ins_labels"].tolist() if x != dataset.tokenizer.pad_token_id]
        assert len(kdr_labels) == len(ins_labels) - 1
