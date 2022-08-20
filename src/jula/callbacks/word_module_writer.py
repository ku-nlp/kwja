import logging
import os
import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
import torch
from jinf import Jinf
from pytorch_lightning.callbacks import BasePredictionWriter
from rhoknp import BasePhrase, Document, Morpheme, Phrase, Sentence
from rhoknp.cohesion import ExophoraReferent, RelTag, RelTagList
from rhoknp.cohesion.discourse_relation import DiscourseRelationTag
from rhoknp.props import DepType, FeatureDict, NamedEntity, NamedEntityCategory, NETagList, SemanticsDict
from rhoknp.units.morpheme import MorphemeAttributes

import jula
from jula.datamodule.datasets.word_dataset import WordDataset
from jula.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJTYPE_CONJFORM_TYPE2ID,
    INDEX2CONJFORM_TYPE,
    INDEX2CONJTYPE_TYPE,
    INDEX2DEPENDENCY_TYPE,
    INDEX2DISCOURSE_RELATION,
    INDEX2POS_TYPE,
    INDEX2SUBPOS_TYPE,
    NE_TAGS,
    POS_SUBPOS_TYPE2ID,
    POS_TYPE2ID,
)
from jula.utils.dependency_parsing import DependencyManager
from jula.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


class WordModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        pred_filename: str = "predict",
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.jinf = Jinf()

        self.destination: Union[Path, TextIO]
        if use_stdout is True:
            self.destination = sys.stdout
        else:
            self.destination = Path(f"{output_dir}/{pred_filename}.knp")
            self.destination.parent.mkdir(exist_ok=True)
            if self.destination.exists():
                os.remove(str(self.destination))

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        sentences: list[Sentence] = []
        dataloaders = trainer.predict_dataloaders
        for prediction in predictions:
            for batch_pred in prediction:
                batch_texts = batch_pred["texts"]
                dataloader_idx: int = batch_pred["dataloader_idx"]
                batch_pos_preds = torch.argmax(batch_pred["word_analysis_pos_logits"], dim=-1)
                batch_subpos_preds = torch.argmax(batch_pred["word_analysis_subpos_logits"], dim=-1)
                batch_conjtype_preds = torch.argmax(batch_pred["word_analysis_conjtype_logits"], dim=-1)
                batch_conjform_preds = torch.argmax(batch_pred["word_analysis_conjform_logits"], dim=-1)
                batch_ne_tag_preds = torch.argmax(batch_pred["ne_logits"], dim=-1)
                batch_word_feature_preds = batch_pred["word_feature_logits"]
                batch_base_phrase_feature_preds = batch_pred["base_phrase_feature_logits"].ge(0.5).long()
                batch_dependency_preds = torch.topk(
                    batch_pred["dependency_logits"],
                    k=4,  # TODO
                    dim=2,
                ).indices
                batch_dependency_type_preds = torch.argmax(batch_pred["dependency_type_logits"], dim=3)
                batch_cohesion_preds = torch.argmax(batch_pred["cohesion_logits"], dim=3)  # (b, rel, word)
                batch_discourse_parsing_preds = torch.argmax(batch_pred["discourse_parsing_logits"], dim=3)
                for (
                    text,
                    pos_preds,
                    subpos_preds,
                    conjtype_preds,
                    conjform_preds,
                    word_feature_preds,
                    ne_tag_preds,
                    base_phrase_feature_preds,
                    dependency_preds,
                    dependency_type_preds,
                    cohesion_preds,
                    discourse_parsing_preds,
                ) in zip(
                    batch_texts,
                    batch_pos_preds.tolist(),
                    batch_subpos_preds.tolist(),
                    batch_conjtype_preds.tolist(),
                    batch_conjform_preds.tolist(),
                    batch_word_feature_preds.tolist(),
                    batch_ne_tag_preds.tolist(),
                    batch_base_phrase_feature_preds.tolist(),
                    batch_dependency_preds.tolist(),
                    batch_dependency_type_preds.tolist(),
                    batch_cohesion_preds.tolist(),
                    batch_discourse_parsing_preds.tolist(),
                ):
                    dataset: WordDataset = dataloaders[dataloader_idx].dataset
                    morphemes = self._create_morphemes(
                        text.split(),
                        pos_preds,
                        subpos_preds,
                        conjtype_preds,
                        conjform_preds,
                    )
                    document = self._chunk_morphemes(morphemes, word_feature_preds)
                    self._add_base_phrase_features(document, base_phrase_feature_preds)
                    self._add_named_entities(document, ne_tag_preds)
                    self._add_dependency(document, dependency_preds, dependency_type_preds, dataset.special_to_index)
                    self._add_cohesion(
                        document,
                        cohesion_preds,
                        dataset.cohesion_rel_types,
                        dataset.exophora_referents,
                        dataset.index_to_special,
                    )
                    doc_id = document.doc_id
                    document = document.reparse()  # reparse to get clauses
                    document.doc_id = doc_id
                    self._add_discourse(document, discourse_parsing_preds)
                    sentences += extract_target_sentences(document)

        output_string = "".join(sentence.to_knp() for sentence in sentences)
        if isinstance(self.destination, Path):
            self.destination.write_text(output_string)
        elif isinstance(self.destination, TextIOBase):
            self.destination.write(output_string)

    def _create_morphemes(
        self,
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
            pos = INDEX2POS_TYPE[pos_index]
            pos_id = POS_TYPE2ID[pos]
            subpos = INDEX2SUBPOS_TYPE[subpos_index]
            subpos_id = POS_SUBPOS_TYPE2ID[pos][subpos]
            conjtype = INDEX2CONJTYPE_TYPE[conjtype_index]
            conjtype_id = conjtype_index
            conjform = INDEX2CONJFORM_TYPE[conjform_index]
            conjform_id = CONJTYPE_CONJFORM_TYPE2ID[conjtype][conjform]
            try:
                lemma = self.jinf(word, conjtype, conjform, "基本形")
            except ValueError as e:
                logger.warning(f"failed to get lemma for {word}: ({e})")
                lemma = word
            attributes = MorphemeAttributes(
                surf=word,
                reading=word,  # TODO
                lemma=lemma,
                pos=pos,
                pos_id=pos_id,
                subpos=subpos,
                subpos_id=subpos_id,
                conjtype=conjtype,
                conjtype_id=conjtype_id,
                conjform=conjform,
                conjform_id=conjform_id,
            )
            morphemes.append(Morpheme(attributes, SemanticsDict(), FeatureDict()))
        return morphemes

    @staticmethod
    def _chunk_morphemes(morphemes: list[Morpheme], word_feature_preds: list[list[float]]) -> Document:
        phrases_buff = []
        base_phrases_buff = []
        morphemes_buff = []
        assert len(morphemes) <= len(word_feature_preds)
        for i, (morpheme, word_feature_pred) in enumerate(zip(morphemes, word_feature_preds)):
            morphemes_buff.append(morpheme)
            # follows jula.utils.constants.WORD_FEATURES
            (
                base_phrase_head_prob,
                base_phrase_end_prob,
                phrase_end_prob,
                declinable_word_surf_head,
                declinable_word_surf_end,
            ) = word_feature_pred
            if base_phrase_head_prob >= 0.5:
                morpheme.features["基本句-主辞"] = True
            # TODO: refactor & set strict condition
            if declinable_word_surf_head >= 0.5:
                morpheme.features["用言表記先頭"] = True
            if declinable_word_surf_end >= 0.5:
                morpheme.features["用言表記末尾"] = True
            # even if base_phrase_end_prob is low, if phrase_end_prob is high enough, create chunk here
            if base_phrase_end_prob >= 0.5 or base_phrase_end_prob + phrase_end_prob >= 1.0:
                base_phrase = BasePhrase(None, None, FeatureDict(), RelTagList(), NETagList(), DiscourseRelationTag())
                base_phrase.morphemes = morphemes_buff
                morphemes_buff = []
                base_phrases_buff.append(base_phrase)
            # even if phrase_end_prob is high, if base_phrase_end_prob is not high enough, do not create chunk here
            if phrase_end_prob >= 0.5 and base_phrase_end_prob + phrase_end_prob >= 1.0:
                phrase = Phrase(None, None, FeatureDict())
                phrase.base_phrases = base_phrases_buff
                base_phrases_buff = []
                phrases_buff.append(phrase)

        # clear buffers
        if morphemes_buff:
            base_phrase = BasePhrase(None, None, FeatureDict(), RelTagList(), NETagList(), DiscourseRelationTag())
            base_phrase.morphemes = morphemes_buff
            base_phrases_buff.append(base_phrase)
        if base_phrases_buff:
            phrase = Phrase(None, None, FeatureDict())
            phrase.base_phrases = base_phrases_buff
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

    @staticmethod
    def _add_named_entities(document: Document, ne_tag_preds: list[int]) -> None:
        for sentence in document.sentences:
            morphemes = sentence.morphemes
            category = ""
            morphemes_buff = []
            for morpheme, ne_tag_pred in zip(morphemes, ne_tag_preds):
                ne_tag = NE_TAGS[ne_tag_pred]
                if ne_tag.startswith("B-"):
                    category = ne_tag[2:]
                    morphemes_buff.append(morpheme)
                elif ne_tag.startswith("I-") and ne_tag[2:] == category:
                    morphemes_buff.append(morpheme)
                else:
                    if morphemes_buff:
                        named_entity = NamedEntity(category=NamedEntityCategory(category), morphemes=morphemes_buff)
                        sentence.named_entities.append(named_entity)
                    category = ""
                    morphemes_buff = []

    @staticmethod
    def _resolve_dependency(base_phrase: BasePhrase, dependency_manager: DependencyManager) -> None:
        src = base_phrase.index
        num_base_phrases = len(base_phrase.sentence.base_phrases)
        for dst in range(src + 1, num_base_phrases):
            dependency_manager.add_edge(src, dst)
            if dependency_manager.has_cycle():
                dependency_manager.remove_edge(src, dst)
            else:
                base_phrase.parent_index = dst
                base_phrase.dep_type = DepType.DEPENDENCY

        for dst in range(src - 1, -1, -1):
            dependency_manager.add_edge(src, dst)
            if dependency_manager.has_cycle():
                dependency_manager.remove_edge(src, dst)
            else:
                base_phrase.parent_index = dst
                base_phrase.dep_type = DepType.DEPENDENCY

        raise RuntimeError("couldn't resolve dependency")

    def _add_dependency(
        self,
        document: Document,
        dependency_preds: list[list[int]],
        dependency_type_preds: list[list[int]],
        special_to_index: dict[str, int],
    ) -> None:
        for sentence in document.sentences:
            base_phrases = sentence.base_phrases
            morpheme_global_index2base_phrase_index = {
                morpheme.global_index: base_phrase.index
                for base_phrase in base_phrases
                for morpheme in base_phrase.morphemes
            }
            morpheme_global_index2base_phrase_index[special_to_index["[ROOT]"]] = -1
            dependency_manager = DependencyManager()
            for base_phrase in base_phrases:
                for parent_morpheme_global_index, dependency_type_id in zip(
                    dependency_preds[base_phrase.head.global_index],
                    dependency_type_preds[base_phrase.head.global_index],
                ):
                    parent_index = morpheme_global_index2base_phrase_index[parent_morpheme_global_index]
                    dependency_manager.add_edge(base_phrase.index, parent_index)
                    if dependency_manager.has_cycle() or (parent_index == -1 and dependency_manager.root):
                        dependency_manager.remove_edge(base_phrase.index, parent_index)
                    else:
                        base_phrase.parent_index = parent_index
                        base_phrase.dep_type = INDEX2DEPENDENCY_TYPE[dependency_type_id]
                        break
                else:
                    if base_phrase == base_phrases[-1] and not dependency_manager.root:
                        base_phrase.parent_index = -1
                        base_phrase.dep_type = DepType.DEPENDENCY
                    else:
                        self._resolve_dependency(base_phrase, dependency_manager)

                if base_phrase.parent_index == -1:
                    base_phrase.phrase.parent_index = -1
                    base_phrase.phrase.dep_type = DepType.DEPENDENCY
                    dependency_manager.root = True
                # base_phrase.phrase.parent_index is None and
                elif base_phrase.phrase != base_phrases[base_phrase.parent_index].phrase:
                    base_phrase.phrase.parent_index = base_phrases[base_phrase.parent_index].phrase.index
                    base_phrase.phrase.dep_type = base_phrase.dep_type

    def _add_cohesion(
        self,
        document: Document,
        cohesion_preds: list[list[int]],
        rel_types: list[str],
        exophora_referents: list[ExophoraReferent],
        index_to_special: dict[int, str],
    ) -> None:
        for base_phrase in document.base_phrases:
            base_phrase.rels = self._to_rels(
                [preds[base_phrase.head.global_index] for preds in cohesion_preds],
                document.morphemes,
                rel_types,
                exophora_referents,
                index_to_special,
            )

    @staticmethod
    def _add_discourse(document: Document, discourse_preds: list[list[int]]) -> None:
        if document.need_clause_tag:
            logger.warning("failed to output clause boundaries")
            return
        for modifier in document.clauses:
            modifier_morpheme_index = modifier.end.morphemes[0].global_index
            preds = []
            for head in document.clauses:
                head_sid = head.sentence.sid
                head_morpheme_index = head.end.morphemes[0].global_index
                head_base_phrase_index = head.end.index
                pred = INDEX2DISCOURSE_RELATION[discourse_preds[modifier_morpheme_index][head_morpheme_index]]
                if pred != "談話関係なし":
                    preds.append(f"{head_sid}/{head_base_phrase_index}/{pred}")
            if preds:
                modifier.end.features["談話関係"] = ";".join(preds)

    @staticmethod
    def _to_rels(
        prediction: list[int],  # (rel)
        morphemes: list[Morpheme],
        rel_types: list[str],
        exophora_referents: list[ExophoraReferent],
        index_to_special: dict[int, str],
    ) -> RelTagList:
        rels = RelTagList()
        assert len(rel_types) == len(prediction)
        for relation, morpheme_index in zip(rel_types, prediction):
            if morpheme_index < 0:
                continue  # non-target phrase
            if 0 <= morpheme_index < len(morphemes):
                # normal
                prediction_bp: BasePhrase = morphemes[morpheme_index].base_phrase
                rels.append(
                    RelTag(
                        type=relation,
                        target=prediction_bp.head.text,
                        sid=prediction_bp.sentence.sid,
                        base_phrase_index=prediction_bp.index,
                        mode=None,
                    )
                )
            elif special_token := index_to_special.get(morpheme_index):
                # exophora
                if special_token in [str(e) for e in exophora_referents]:  # exclude [NULL], [NA], and [ROOT]
                    rels.append(
                        RelTag(
                            type=relation,
                            target=special_token,
                            sid=None,
                            base_phrase_index=None,
                            mode=None,
                        )
                    )
            else:
                raise ValueError(f"invalid morpheme index: {morpheme_index} in {morphemes[0].document.doc_id}")

        return rels

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass
