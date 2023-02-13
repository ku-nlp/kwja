import tempfile
import textwrap

import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from kwja.callbacks.typo_module_writer import TypoModuleWriter
from kwja.datamodule.datasets.typo_inference_dataset import TypoInferenceDataset
from kwja.utils.constants import TYPO_CORR_OP_TAG2TOKEN


class MockTrainer:
    def __init__(self, predict_dataloaders):
        self.predict_dataloaders = predict_dataloaders


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer = AutoTokenizer.from_pretrained(
            "ku-nlp/roberta-base-japanese-char-wwm",
            do_word_tokenize=False,
            additional_special_tokens=["<k>", "<d>", "<_>", "<dummy>"],
        )
        _ = TypoModuleWriter(
            tmp_dir,
            extended_vocab_path="tests/datamodule/datasets/typo_files/extended_vocab.txt",
            confidence_threshold=0.9,
            tokenizer=tokenizer,
        )


def test_write_on_batch_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer = AutoTokenizer.from_pretrained(
            "ku-nlp/roberta-base-japanese-char-wwm",
            do_word_tokenize=False,
            additional_special_tokens=["<k>", "<d>", "<_>", "<dummy>"],
        )
        writer = TypoModuleWriter(
            tmp_dir,
            extended_vocab_path="tests/datamodule/datasets/typo_files/extended_vocab.txt",
            confidence_threshold=0.9,
            tokenizer=tokenizer,
        )
        text = "今日がも晴れ"
        dataset = TypoInferenceDataset(texts=ListConfig([text]), tokenizer=tokenizer, max_seq_length=len(text) + 3)
        trainer = MockTrainer([DataLoader(dataset)])

        kdr_logits = torch.full((1, 9, len(writer.token2token_id)), float("-inf"), dtype=torch.float)
        k_token = TYPO_CORR_OP_TAG2TOKEN["K"]
        d_token = TYPO_CORR_OP_TAG2TOKEN["D"]
        kdr_logits[0][0][writer.token2token_id[k_token]] = 1.0  # keep: [CLS]
        kdr_logits[0][1][writer.token2token_id[k_token]] = 1.0  # keep: 今
        kdr_logits[0][2][writer.token2token_id[k_token]] = 1.0  # keep: 日
        kdr_logits[0][3][writer.token2token_id["は"]] = 1.0  # replace: が -> は
        kdr_logits[0][4][writer.token2token_id[d_token]] = 1.0  # delete: も ->
        kdr_logits[0][5][writer.token2token_id[k_token]] = 1.0  # keep: 晴
        kdr_logits[0][6][writer.token2token_id[k_token]] = 1.0  # keep: れ
        kdr_logits[0][7][writer.token2token_id[k_token]] = 1.0  # insert: <dummy> -> だ
        kdr_logits[0][8][writer.token2token_id[k_token]] = 1.0  # keep: [SEP]

        ins_logits = torch.full((1, 9, len(writer.token2token_id)), float("-inf"), dtype=torch.float)
        __token = TYPO_CORR_OP_TAG2TOKEN["_"]
        ins_logits[0][0][writer.token2token_id[__token]] = 1.0  # keep: [CLS]
        ins_logits[0][1][writer.token2token_id[__token]] = 1.0  # keep: 今
        ins_logits[0][2][writer.token2token_id[__token]] = 1.0  # keep: 日
        ins_logits[0][3][writer.token2token_id[__token]] = 1.0  # replace: が -> は
        ins_logits[0][4][writer.token2token_id[__token]] = 1.0  # delete: も ->
        ins_logits[0][5][writer.token2token_id[__token]] = 1.0  # keep: 晴
        ins_logits[0][6][writer.token2token_id[__token]] = 1.0  # keep: れ
        ins_logits[0][7][writer.token2token_id["だ"]] = 1.0  # insert: <dummy> -> だ
        ins_logits[0][8][writer.token2token_id[__token]] = 1.0  # keep: [SEP]

        kdr_probabilities = kdr_logits.softmax(dim=2)
        kdr_max_probabilities, kdr_predictions = kdr_probabilities.max(dim=2)
        ins_probabilities = ins_logits.softmax(dim=2)
        ins_max_probabilities, ins_predictions = ins_probabilities.max(dim=2)
        prediction = {
            "example_ids": torch.tensor([0], dtype=torch.long),
            "kdr_probabilities": kdr_max_probabilities,
            "kdr_predictions": kdr_predictions,
            "ins_probabilities": ins_max_probabilities,
            "ins_predictions": ins_predictions,
        }

        writer.write_on_batch_end(trainer, ..., prediction, ..., ..., 0, 0)
        with open(writer.destination) as f:
            assert f.read() == textwrap.dedent(
                """\
                今日は晴れだ
                EOD
                """
            )
