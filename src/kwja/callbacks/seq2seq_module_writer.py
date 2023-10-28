from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pytorch_lightning as pl
from rhoknp import Sentence
from transformers import PreTrainedTokenizerFast

import kwja
from kwja.callbacks.base_module_writer import BaseModuleWriter
from kwja.datamodule.datasets import Seq2SeqDataset, Seq2SeqInferenceDataset
from kwja.datamodule.examples import Seq2SeqExample, Seq2SeqInferenceExample
from kwja.utils.seq2seq_format import Seq2SeqFormatter


class Seq2SeqModuleWriter(BaseModuleWriter):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, destination: Optional[Union[str, Path]] = None) -> None:
        super().__init__(destination=destination)
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.formatter: Seq2SeqFormatter = Seq2SeqFormatter(tokenizer)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(trainer.predict_dataloaders, dict):
            dataloader = list(trainer.predict_dataloaders.values())[dataloader_idx]
        else:
            dataloader = trainer.predict_dataloaders[dataloader_idx]
        dataset: Union[Seq2SeqDataset, Seq2SeqInferenceDataset] = dataloader.dataset

        output_string = ""
        for example_id, seq2seq_predictions in zip(*[v.tolist() for v in prediction.values()]):
            example: Union[Seq2SeqExample, Seq2SeqInferenceExample] = dataset.examples[example_id]
            if len(example.surfs) == 0:
                continue

            decoded: str = self.tokenizer.decode(
                [x for x in seq2seq_predictions if x not in {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}],
                skip_special_tokens=False,
            )
            seq2seq_format: Sentence = self.formatter.format_to_sent(decoded.replace(" ", ""))
            seq2seq_format.sid = example.sid
            seq2seq_format.misc_comment = f"kwja:{kwja.__version__}"
            output_string += seq2seq_format.to_jumanpp()
        self.write_output_string(output_string)
