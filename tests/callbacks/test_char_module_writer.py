import tempfile

import torch
from torch.utils.data import DataLoader

from jula.callbacks.char_module_writer import CharModuleWriter
from jula.datamodule.datasets.char_inference_dataset import CharInferenceDataset


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _ = CharModuleWriter(tmp_dir)


class MockTrainer:
    def __init__(self, predict_dataloaders):
        self.predict_dataloaders = predict_dataloaders


def test_write_on_epoch_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = CharModuleWriter(tmp_dir)
        dataset = CharInferenceDataset(
            texts=["今日は晴れ"],
            model_name_or_path="cl-tohoku/bert-base-japanese-char",
        )
        trainer = MockTrainer([DataLoader(dataset)])
        input_ids = dataset.tokenizer("今日は晴れ", return_tensors="pt")["input_ids"]
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
                    "dataloader_idx": 0,
                    "input_ids": input_ids,
                    "word_segmenter_logits": word_segmenter_logits,
                }
            ]
        ]
        writer.write_on_epoch_end(trainer, ..., predictions)
        with open(writer.destination) as f:
            assert f.read().strip() == "今日 は 晴れ"
