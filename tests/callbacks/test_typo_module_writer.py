import tempfile
import textwrap

import torch

from kwja.callbacks.typo_module_writer import TypoModuleWriter
from kwja.utils.constants import TYPO_OPN2TOKEN


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = TypoModuleWriter(
            tmp_dir,
            extended_vocab_path="tests/datamodule/datasets/typo_files/extended_vocab.txt",
            confidence_threshold=0.9,
            tokenizer_kwargs={
                "do_word_tokenize": False,
                "additional_special_tokens": ["<k>", "<d>", "<_>", "<dummy>"],
            },
        )


def test_write_on_batch_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = TypoModuleWriter(
            tmp_dir,
            extended_vocab_path="tests/datamodule/datasets/typo_files/extended_vocab.txt",
            confidence_threshold=0.9,
            tokenizer_kwargs={
                "do_word_tokenize": False,
                "additional_special_tokens": ["<k>", "<d>", "<_>", "<dummy>"],
            },
        )

        # tokenizer = writer.tokenizer
        texts = ["今日がも晴れ"]

        kdr_logits = torch.full((1, 9, len(writer.opn2id)), float("-inf"), dtype=torch.float)
        kdr_logits[0][0][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: [CLS]
        kdr_logits[0][1][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: 今
        kdr_logits[0][2][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: 日
        kdr_logits[0][3][writer.opn2id["は"]] = 1.0  # replace: が -> は
        kdr_logits[0][4][writer.opn2id[TYPO_OPN2TOKEN["D"]]] = 1.0  # delete: も ->
        kdr_logits[0][5][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: 晴
        kdr_logits[0][6][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: れ
        kdr_logits[0][7][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # insert: <dummy> -> だ
        kdr_logits[0][8][writer.opn2id[TYPO_OPN2TOKEN["K"]]] = 1.0  # keep: [SEP]

        ins_logits = torch.full((1, 9, len(writer.opn2id)), float("-inf"), dtype=torch.float)
        ins_logits[0][0][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: [CLS]
        ins_logits[0][1][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: 今
        ins_logits[0][2][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: 日
        ins_logits[0][3][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # replace: が -> は
        ins_logits[0][4][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # delete: も ->
        ins_logits[0][5][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: 晴
        ins_logits[0][6][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: れ
        ins_logits[0][7][writer.opn2id["だ"]] = 1.0  # insert: <dummy> -> だ
        ins_logits[0][8][writer.opn2id[TYPO_OPN2TOKEN["_"]]] = 1.0  # keep: [SEP]

        kdr_probs = torch.softmax(kdr_logits[:, 1:, :], dim=-1)
        kdr_values, kdr_indices = torch.max(kdr_probs, dim=-1)
        ins_probs = torch.softmax(ins_logits[:, 1:, :], dim=-1)
        ins_values, ins_indices = torch.max(ins_probs, dim=-1)
        prediction = {
            "texts": texts,
            "kdr_values": kdr_values,
            "kdr_indices": kdr_indices,
            "ins_values": ins_values,
            "ins_indices": ins_indices,
        }
        writer.write_on_batch_end(..., ..., prediction, ..., ..., ..., ...)
        with open(writer.destination) as f:
            assert f.read() == textwrap.dedent(
                """\
                今日は晴れだ
                EOD
                """
            )
