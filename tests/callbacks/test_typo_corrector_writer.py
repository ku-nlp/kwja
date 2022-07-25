import tempfile

import torch

from jula.callbacks.typo_corrector_writer import TypoCorrectorWriter
from jula.utils.constants import TYPO_OPN2TOKEN


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = TypoCorrectorWriter(
            tmp_dir,
            extended_vocab_path="tests/datamodule/datasets/typo_files/extended_vocab.txt",
            tokenizer_kwargs={
                "do_word_tokenize": False,
                "additional_special_tokens": ["<k>", "<d>", "<_>", "<dummy>"],
            },
        )


def test_write_on_epoch_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = TypoCorrectorWriter(
            tmp_dir,
            extended_vocab_path="tests/datamodule/datasets/typo_files/extended_vocab.txt",
            tokenizer_kwargs={
                "do_word_tokenize": False,
                "additional_special_tokens": ["<k>", "<d>", "<_>", "<dummy>"],
            },
        )

        # tokenizer = writer.tokenizer
        texts = ["今日がも晴れ"]

        kdr_logits = torch.zeros(1, 9, len(writer.opn2id), dtype=torch.float)
        kdr_logits[0][0][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: [CLS]
        kdr_logits[0][1][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: 今
        kdr_logits[0][2][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: 日
        kdr_logits[0][3][writer.opn2id["は"]] = 1.0  # replace: が -> は
        kdr_logits[0][4][writer.opn2id[TYPO_OPN2TOKEN["D"]]] = 1.0  # delete: も ->
        kdr_logits[0][5][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: 晴
        kdr_logits[0][6][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: れ
        kdr_logits[0][7][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # insert: <dummy> -> だ
        kdr_logits[0][8][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: [SEP]

        ins_logits = torch.zeros(1, 9, len(writer.opn2id), dtype=torch.float)
        ins_logits[0][0][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: [CLS]
        ins_logits[0][1][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: 今
        ins_logits[0][2][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: 日
        ins_logits[0][3][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # replace: が -> は
        ins_logits[0][4][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # delete: も ->
        ins_logits[0][5][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: 晴
        ins_logits[0][6][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: れ
        ins_logits[0][7][writer.opn2id["だ"]] = 1.0  # insert: <dummy> -> だ
        ins_logits[0][8][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: [SEP]

        predictions = [
            [
                {
                    "texts": texts,
                    "kdr_logits": kdr_logits,
                    "ins_logits": ins_logits,
                }
            ]
        ]
        writer.write_on_epoch_end(..., ..., predictions)
        with open(writer.output_path) as f:
            assert f.read().strip() == "今日は晴れだ"
