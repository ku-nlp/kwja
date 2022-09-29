import tempfile

import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader

import kwja
from kwja.callbacks.char_module_writer import CharModuleWriter
from kwja.datamodule.datasets.char_inference_dataset import CharInferenceDataset


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
            texts=ListConfig(["今日は晴れだぁ"]),
            document_split_stride=1,
            model_name_or_path="ku-nlp/roberta-base-japanese-char-wwm",
            doc_id_prefix="test",
        )
        trainer = MockTrainer([DataLoader(dataset)])
        input_ids = dataset.tokenizer("今日は晴れだぁ", return_tensors="pt")["input_ids"]
        word_segmenter_logits = torch.tensor(
            [
                [
                    [0.0, 0.0],  # CLS
                    [1.0, 0.0],  # 今
                    [0.0, 1.0],  # 日
                    [1.0, 0.0],  # は
                    [1.0, 0.0],  # 晴
                    [0.0, 1.0],  # れ
                    [1.0, 0.0],  # だ
                    [0.0, 1.0],  # ぁ
                    [0.0, 0.0],  # SEP
                ]
            ],
            dtype=torch.float,
        )
        word_normalizer_logits = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # CLS
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 今
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 日
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # は
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 晴
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # れ
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # だ
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # ぁ
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # SEP
                ]
            ],
            dtype=torch.float,
        )
        predictions = [
            [
                {
                    "example_ids": [0],
                    "dataloader_idx": 0,
                    "input_ids": input_ids,
                    "word_segmenter_logits": word_segmenter_logits,
                    "word_normalizer_logits": word_normalizer_logits,
                }
            ]
        ]
        writer.write_on_epoch_end(trainer, ..., predictions)
        assert writer.destination.read_text() == f"# S-ID:test-0-0 kwja:{kwja.__version__}\n今日 は 晴れ だ\n"
