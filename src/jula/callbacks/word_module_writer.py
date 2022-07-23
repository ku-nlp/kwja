import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from rhoknp import BasePhrase, Document, Morpheme, Phrase, Sentence
from rhoknp.rel import ExophoraReferent
from rhoknp.units.morpheme import MorphemeAttributes
from rhoknp.units.utils import Features, Rel, Rels, Semantics
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import jula
from jula.utils.constants import (
    BASE_PHRASE_FEATURES,
    INDEX2CONJFORM_TYPE,
    INDEX2CONJTYPE_TYPE,
    INDEX2DEPENDENCY_TYPE,
    INDEX2POS_TYPE,
    INDEX2SUBPOS_TYPE,
)


class WordModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        tokenizer_kwargs: dict = None,
        max_seq_length: int = 512,
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.destination: Union[Path, TextIO]
        if use_stdout is True:
            self.destination = sys.stdout
        else:
            self.destination = Path(f"{output_dir}/{pred_filename}.knp")
            self.destination.parent.mkdir(exist_ok=True)
            if self.destination.exists():
                os.remove(str(self.destination))

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **hydra.utils.instantiate(tokenizer_kwargs or {}, _convert_="partial"),
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        # cohesion analysis settings
        # TODO: fix hard coding
        self.relations = ["ガ", "ヲ", "ニ", "ガ２", "ノ", "="]
        self.exophora_referents: list[ExophoraReferent] = [ExophoraReferent(s) for s in ("著者", "読者", "不特定:人", "不特定:物")]
        self.special_tokens: list[str] = [str(e) for e in self.exophora_referents] + [
            "[NULL]",
            "[NA]",
            "[ROOT]",
        ]
        self.special_to_index: dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        self.index_to_special: dict[int, str] = {v: k for k, v in self.special_to_index.items()}

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        documents: list[Document] = []
        for prediction in predictions:
            for batch_pred in prediction:
                batch_texts = batch_pred["texts"]
                batch_pos_preds = torch.argmax(batch_pred["word_analysis_pos_logits"], dim=-1)
                batch_subpos_preds = torch.argmax(batch_pred["word_analysis_subpos_logits"], dim=-1)
                batch_conjtype_preds = torch.argmax(batch_pred["word_analysis_conjtype_logits"], dim=-1)
                batch_conjform_preds = torch.argmax(batch_pred["word_analysis_conjform_logits"], dim=-1)
                batch_word_feature_preds = batch_pred["word_feature_logits"].ge(0.5).long()
                batch_base_phrase_feature_preds = batch_pred["base_phrase_feature_logits"].ge(0.5).long()
                batch_dependency_preds = torch.topk(
                    batch_pred["dependency_logits"],
                    # pl_module.hparams.k,  # TODO: move to WordModuleWriter's config or argument
                    k=1,
                    dim=2,
                ).indices
                batch_dependency_type_preds = torch.argmax(batch_pred["dependency_type_logits"], dim=3)
                batch_cohesion_preds = torch.argmax(batch_pred["cohesion_logits"], dim=3)  # (b, rel, word)
                for (
                    text,
                    pos_preds,
                    subpos_preds,
                    conjtype_preds,
                    conjform_preds,
                    word_feature_preds,
                    base_phrase_feature_preds,
                    dependency_preds,
                    dependency_type_preds,
                    cohesion_preds,
                ) in zip(
                    batch_texts,
                    batch_pos_preds.tolist(),
                    batch_subpos_preds.tolist(),
                    batch_conjtype_preds.tolist(),
                    batch_conjform_preds.tolist(),
                    batch_word_feature_preds.tolist(),
                    batch_base_phrase_feature_preds.tolist(),
                    batch_dependency_preds.tolist(),
                    batch_dependency_type_preds.tolist(),
                    batch_cohesion_preds.tolist(),
                ):
                    morphemes = self._create_morphemes(
                        text.split(),
                        pos_preds,
                        subpos_preds,
                        conjtype_preds,
                        conjform_preds,
                    )
                    document = self._chunk_morphemes(morphemes, word_feature_preds)
                    self._add_base_phrase_features(document, base_phrase_feature_preds)
                    self._add_dependency(document, dependency_preds, dependency_type_preds)
                    self._add_cohesion(document, cohesion_preds)
                    documents.append(document)

        output_string = "".join(doc.to_knp() for doc in documents)
        if isinstance(self.destination, Path):
            self.destination.write_text(output_string)
        elif isinstance(self.destination, TextIO):
            self.destination.write(output_string)

    @staticmethod
    def _create_morphemes(
        words: list[str],
        pos_preds: list[int],
        subpos_preds: list[int],
        conjtype_preds: list[int],
        conjform_preds: list[int],
    ) -> list[Morpheme]:
        morphemes = []
        for word, pos_index, subpos_index, conjtype_index, conjform_index in zip(
            words, pos_preds, subpos_preds, conjtype_preds, conjform_preds
        ):
            attributes = MorphemeAttributes(
                surf=word,
                reading=word,  # TODO
                lemma=word,  # TODO
                pos=INDEX2POS_TYPE[pos_index],
                pos_id=0,  # TODO
                subpos=INDEX2SUBPOS_TYPE[subpos_index],
                subpos_id=0,  # TODO
                conjtype=INDEX2CONJTYPE_TYPE[conjtype_index],
                conjtype_id=0,  # TODO
                conjform=INDEX2CONJFORM_TYPE[conjform_index],
                conjform_id=0,  # TODO
            )
            morphemes.append(Morpheme(attributes, Semantics({}), Features()))
        return morphemes

    @staticmethod
    def _chunk_morphemes(morphemes: list[Morpheme], word_feature_preds: list[list[int]]) -> Document:
        phrases_buff = []
        base_phrases_buff = []
        morphemes_buff = []
        assert len(morphemes) <= len(word_feature_preds)
        for i, (morpheme, word_feature_pred) in enumerate(zip(morphemes, word_feature_preds)):
            morphemes_buff.append(morpheme)
            # follows WORD_FEATURES
            is_base_phrase_head, is_base_phrase_end, is_phrase_end = map(bool, word_feature_pred)
            if is_base_phrase_head is True:
                morpheme.features["基本句-主辞"] = True
            if is_base_phrase_end is True:
                base_phrase = BasePhrase(None, None, Features(), Rels())
                base_phrase.morphemes = morphemes_buff
                morphemes_buff = []
                base_phrases_buff.append(base_phrase)
            if is_phrase_end is True:
                phrase = Phrase(None, None, Features())
                phrase.base_phrases = base_phrases_buff
                base_phrases_buff = []
                phrases_buff.append(phrase)
        sentence = Sentence()
        sentence.sid = "1"
        sentence.misc_comment = f"jula:{jula.__version__}"
        sentence.phrases = phrases_buff
        # TODO: support document with multiple sentences
        return Document.from_sentences([sentence])

    @staticmethod
    def _add_base_phrase_features(document: Document, base_phrase_feature_preds: list[list[int]]) -> None:
        for base_phrase in document.base_phrases:
            for feature, pred in zip(BASE_PHRASE_FEATURES, base_phrase_feature_preds[base_phrase.head.global_index]):
                if pred == 1:
                    k, *vs = feature.split(":")
                    base_phrase.features[k] = ":".join(vs) or True

    def _add_dependency(
        self, document: Document, dependency_preds: list[list[int]], dependency_type_preds: list[list[int]]
    ) -> None:
        morphemes = document.morphemes
        for base_phrase in document.base_phrases:
            parent_morpheme_index = dependency_preds[base_phrase.head.global_index][0]
            dependency_type_id = dependency_type_preds[base_phrase.head.global_index][0]
            if 0 <= parent_morpheme_index < len(morphemes):
                base_phrase.parent_index = morphemes[parent_morpheme_index].base_phrase.index
            elif parent_morpheme_index == self.special_to_index["[ROOT]"]:
                base_phrase.parent_index = -1
            base_phrase.dep_type = INDEX2DEPENDENCY_TYPE[dependency_type_id]

    def _add_cohesion(self, document: Document, cohesion_preds: list[list[int]]) -> None:
        for base_phrase in document.base_phrases:
            base_phrase.rels = self._to_rels(
                [preds[base_phrase.head.global_index] for preds in cohesion_preds], document.morphemes
            )

    def _to_rels(
        self,
        prediction: list[int],  # (rel)
        morphemes: list[Morpheme],
    ) -> Rels:
        rels: Rels = Rels([])
        assert len(self.relations) == len(prediction)
        for relation, morpheme_index in zip(self.relations, prediction):
            if morpheme_index < 0:
                continue  # non-target phrase
            if 0 <= morpheme_index < len(morphemes):
                # normal
                prediction_bp: BasePhrase = morphemes[morpheme_index].base_phrase
                rels.append(
                    Rel(
                        type=relation,
                        target=prediction_bp.head.text,
                        sid=prediction_bp.sentence.sid,
                        base_phrase_index=prediction_bp.index,
                        mode=None,
                    )
                )
            elif special_token := self.index_to_special.get(morpheme_index):
                # exophora
                if special_token in [str(e) for e in self.exophora_referents]:  # exclude [NULL], [NA], and [ROOT]
                    rels.append(
                        Rel(
                            type=relation,
                            target=special_token,
                            sid=None,
                            base_phrase_index=None,
                            mode=None,
                        )
                    )
            else:
                raise ValueError(f"invalid morpheme index: {morpheme_index}")

        return rels
