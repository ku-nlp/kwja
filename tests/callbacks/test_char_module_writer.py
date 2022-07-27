import tempfile

import torch

from jula.callbacks.char_module_writer import WordModuleWriter


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = WordModuleWriter(tmp_dir)


def test_write_on_epoch_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = WordModuleWriter(tmp_dir)

        tokenizer = writer.tokenizer
        input_ids = tokenizer("今日は晴れ", return_tensors="pt")["input_ids"]
        word_segmenter_logits = torch.tensor(
            [
                [
                    [0.0, 0.0],  # CLS
                    [1.0, 0.0],  # 今
                    [0.0, 1.0],  # 日
                    [1.0, 0.0],  # は
                    [1.0, 0.0],  # 晴
                    [0.0, 1.0],  # れ
                    [0.0, 0.0],  # SEP
                ]
            ],
            dtype=torch.float,
        )
        predictions = [
            [
                {
                    "input_ids": input_ids,
                    "word_segmenter_logits": word_segmenter_logits,
                }
            ]
        ]
        writer.write_on_epoch_end(..., ..., predictions)
        with open(writer.output_path) as f:
            assert f.read().strip() == "今日 は 晴れ"
