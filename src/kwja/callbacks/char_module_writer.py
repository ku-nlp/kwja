from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import lightning as L

import kwja
from kwja.callbacks.base_module_writer import BaseModuleWriter
from kwja.callbacks.utils import convert_char_predictions_into_tags, set_morphemes, set_sentences
from kwja.datamodule.datasets import CharDataset, CharInferenceDataset
from kwja.datamodule.examples import CharExample, CharInferenceExample
from kwja.utils.sub_document import to_orig_doc_id


class CharModuleWriter(BaseModuleWriter):
    def __init__(self, destination: Optional[Union[str, Path]] = None) -> None:
        super().__init__(destination=destination)
        self.prev_doc_id = ""
        self.prev_sid = 0

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,  # noqa: ARG002
        prediction: Any,
        batch_indices: Optional[Sequence[int]],  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,
    ) -> None:
        if isinstance(trainer.predict_dataloaders, dict):
            dataloader = list(trainer.predict_dataloaders.values())[dataloader_idx]
        else:
            dataloader = trainer.predict_dataloaders[dataloader_idx]
        dataset: Union[CharDataset, CharInferenceDataset] = dataloader.dataset

        special_ids = {
            getattr(dataset.tokenizer, f"{prefix}_token_id") for prefix in ["bos", "eos", "sep", "pad", "cls", "mask"]
        }

        for example_id, sent_segmentation_predictions, word_segmentation_predictions, word_norm_op_predictions in zip(
            *[v.tolist() for v in prediction.values()]
        ):
            example: Union[CharExample, CharInferenceExample] = dataset.examples[example_id]
            assert example.doc_id is not None, "doc_id isn't set"
            document = dataset.doc_id2document.pop(example.doc_id)

            sent_segmentation_tags, word_segmentation_tags, word_norm_op_tags = convert_char_predictions_into_tags(
                sent_segmentation_predictions,
                word_segmentation_predictions,
                word_norm_op_predictions,
                [i for i, input_id in enumerate(example.encoding.input_ids) if input_id not in special_ids],
            )
            set_sentences(document, sent_segmentation_tags)
            set_morphemes(document, word_segmentation_tags, word_norm_op_tags)

            orig_doc_id = to_orig_doc_id(document.doc_id)
            if orig_doc_id != self.prev_doc_id:
                self.prev_doc_id = orig_doc_id
                self.prev_sid = 1  # 1-origin

            output_string = ""
            # Every sentence is a target sentence because document_split_stride is always -1
            for sentence in document.sentences:
                output_string += f"# S-ID:{orig_doc_id}-{self.prev_sid} kwja:{kwja.__version__}\n"
                output_string += sentence.to_jumanpp()
                self.prev_sid += 1
            self.write_output_string(output_string)
