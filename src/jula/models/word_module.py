from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule


class WordModule(LightningModule):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
