import copy
from typing import Any, Dict

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig, OmegaConf


class BaseModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

    def configure_optimizers(self):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ("bias", "LayerNorm.weight")
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "name": "decay",
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=optimizer_grouped_parameters, _convert_="partial"
        )
        total_steps = self.trainer.estimated_stepping_batches
        if hasattr(self.hparams.scheduler, "num_training_steps"):
            self.hparams.scheduler.num_training_steps = total_steps
        lr_scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        hparams: DictConfig = copy.deepcopy(checkpoint["hyper_parameters"])
        OmegaConf.set_struct(hparams, False)
        if self.hparams.ignore_hparams_on_save:
            hparams = filter_dict_items(hparams, self.hparams.hparams_to_ignore_on_save)
        checkpoint["hyper_parameters"] = hparams


def filter_dict_items(item: DictConfig, keys_to_ignore: ListConfig) -> DictConfig:
    """Filter out dictionary items whose key is in keys_to_ignore recursively."""
    for key, value in item.items():
        ignore = False
        for key_to_ignore in keys_to_ignore:
            if isinstance(key_to_ignore, str) and key == key_to_ignore:
                ignore = True
                break
            elif isinstance(key_to_ignore, (dict, DictConfig)) and key in key_to_ignore:
                item[key] = filter_dict_items(value, key_to_ignore[key])
                break
        if ignore is True:
            del item[key]
    return item
