import logging
from typing import Dict, Union

import hydra
import transformers.utils.logging as hf_logging
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch.utils.data import ConcatDataset

from kwja.datamodule.datamodule import DataModule
from kwja.datamodule.datasets.word_dataset import WordDataset
from kwja.utils.constants import IGNORE_INDEX
from kwja.utils.reading_prediction import UNK_ID

hf_logging.set_verbosity(hf_logging.ERROR)
logging.getLogger("rhoknp").setLevel(logging.WARNING)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


def get_word_dataset_stats(dataset: WordDataset) -> Dict[str, Union[int, float]]:
    stats: Dict[str, Union[int, float]] = {"num_examples": len(dataset)}

    # Reading predictions
    total_num_reading_labels = 0
    total_num_unk_reading_labels = 0

    # Iterate over the dataset
    for index in range(len(dataset)):
        example = dataset[index]

        # Reading predictions
        reading_ids = example["reading_ids"]
        num_reading_labels = reading_ids.ne(IGNORE_INDEX).sum().item()
        assert type(num_reading_labels) == int, "type of num_reading_labels is invalid"
        total_num_reading_labels += num_reading_labels
        num_unk_reading_labels = reading_ids.eq(UNK_ID).sum().item()
        assert type(num_unk_reading_labels) == int, "type of num_unk_reading_labels is invalid"
        total_num_unk_reading_labels += num_unk_reading_labels

    # Reading predictions
    stats["reading_prediction/num_labels"] = total_num_reading_labels
    stats["reading_prediction/num_unk_labels"] = total_num_unk_reading_labels
    stats["reading_prediction/ratio_unk_labels"] = total_num_unk_reading_labels / total_num_reading_labels

    return stats


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    load_dotenv()

    datamodule = DataModule(cfg=cfg.datamodule)
    datamodule.setup("fit")
    datamodule.setup("test")

    if cfg.config_name in cfg.module.word:
        for corpus_index, corpus in enumerate(cfg.datamodule.train.keys()):
            assert type(datamodule.train_dataset) == ConcatDataset
            dataset = datamodule.train_dataset.datasets[corpus_index]
            assert isinstance(dataset, WordDataset)
            print({f"{corpus}/train/{k}": v for k, v in get_word_dataset_stats(dataset).items()})
        for corpus in cfg.datamodule.valid.keys():
            dataset = datamodule.valid_datasets[corpus]
            assert isinstance(dataset, WordDataset)
            print({f"{corpus}/valid/{k}": v for k, v in get_word_dataset_stats(dataset).items()})
        for corpus in cfg.datamodule.test.keys():
            dataset = datamodule.test_datasets[corpus]
            assert isinstance(dataset, WordDataset)
            print({f"{corpus}/test/{k}": v for k, v in get_word_dataset_stats(dataset).items()})
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
