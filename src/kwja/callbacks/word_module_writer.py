import copy
import itertools
import logging
import os
import sys
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO, Union

import pytorch_lightning as pl
import torch
from BetterJSONStorage import BetterJSONStorage
from jinf import Jinf
from pytorch_lightning.callbacks import BasePredictionWriter
from rhoknp import BasePhrase, Document, Morpheme, Phrase, Sentence
from rhoknp.cohesion import ExophoraReferent, RelTag, RelTagList
from rhoknp.props import DepType, FeatureDict, NamedEntity, NamedEntityCategory, SemanticsDict
from rhoknp.units.morpheme import MorphemeAttributes
from tinydb import Query, TinyDB
from tinydb.middlewares import CachingMiddleware

from kwja.datamodule.datasets import WordDataset, WordInferenceDataset
from kwja.datamodule.datasets.word_dataset import WordExampleSet
from kwja.datamodule.datasets.word_inference_dataset import WordInferenceExample
from kwja.datamodule.examples import CohesionTask
from kwja.datamodule.extractors.base import Extractor
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TYPES,
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
    SUBPOS_TYPES,
)
from kwja.utils.dependency_parsing import DependencyManager
from kwja.utils.reading import get_reading2id, get_word_level_readings
from kwja.utils.sub_document import extract_target_sentences

logger = logging.getLogger(__name__)


class WordModuleWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        reading_resource_path: str,
        jumandic_path: str,
        ambig_surf_specs: list[dict[str, str]],
        pred_filename: str = "predict",
        use_stdout: bool = False,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.reading_resource_path = Path(reading_resource_path)
        reading2id = get_reading2id(str(self.reading_resource_path / "vocab.txt"))
        self.id2reading = {v: k for k, v in reading2id.items()}
        self.jinf = Jinf()
        self.jumandic_path = Path(jumandic_path)
        self.jumandic = TinyDB(
            self.jumandic_path / "jumandic.db",
            ensure_ascii=False,
            access_mode="r",
            storage=CachingMiddleware(BetterJSONStorage),
        )
        self.ambig_surf_specs = ambig_surf_specs

        self.destination: Union[Path, TextIO]
        if use_stdout is True:
            self.destination = sys.stdout
        else:
            self.destination = Path(f"{output_dir}/{pred_filename}.knp")
            self.destination.parent.mkdir(exist_ok=True, parents=True)
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
                batch_tokens = batch_pred["tokens"]
                batch_example_ids = batch_pred["example_ids"]
                dataloader_idx: int = batch_pred["dataloader_idx"]
                dataset: Union[WordDataset, WordInferenceDataset] = dataloaders[dataloader_idx].dataset
                batch_reading_subword_map = batch_pred["reading_subword_map"]
                batch_reading_preds = torch.argmax(batch_pred["reading_prediction_logits"], dim=-1)
                (
                    batch_pos_preds,
                    batch_subpos_preds,
                    batch_conjtype_preds,
                    batch_conjform_preds,
                ) = self._get_mrph_type_preds(
                    pos_logits=batch_pred["word_analysis_pos_logits"],
                    subpos_logits=batch_pred["word_analysis_subpos_logits"],
                    conjtype_logits=batch_pred["word_analysis_conjtype_logits"],
                    conjform_logits=batch_pred["word_analysis_conjform_logits"],
                )
                batch_ne_tag_preds = torch.argmax(batch_pred["ne_logits"], dim=-1)
                batch_word_feature_preds = batch_pred["word_feature_logits"]
                batch_base_phrase_feature_preds = batch_pred["base_phrase_feature_logits"]
                batch_dependency_preds = torch.topk(
                    batch_pred["dependency_logits"],
                    k=4,  # TODO
                    dim=2,
                ).indices
                batch_dependency_type_preds = torch.argmax(batch_pred["dependency_type_logits"], dim=3)
                batch_cohesion_preds = torch.argmax(batch_pred["cohesion_logits"], dim=3)  # (b, rel, word)
                batch_discourse_parsing_preds = torch.argmax(batch_pred["discourse_parsing_logits"], dim=3)
                for (
                    tokens,
                    example_id,
                    reading_subword_map,
                    reading_preds,
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
                    batch_tokens,
                    batch_example_ids,
                    batch_reading_subword_map.tolist(),
                    batch_reading_preds.tolist(),
                    batch_pos_preds,
                    batch_subpos_preds,
                    batch_conjtype_preds,
                    batch_conjform_preds,
                    batch_word_feature_preds.tolist(),
                    batch_ne_tag_preds.tolist(),
                    batch_base_phrase_feature_preds.tolist(),
                    batch_dependency_preds.tolist(),
                    batch_dependency_type_preds.tolist(),
                    batch_cohesion_preds.tolist(),
                    batch_discourse_parsing_preds.tolist(),
                ):
                    example: Union[WordExampleSet, WordInferenceExample] = dataset.examples[example_id]
                    doc_id = example.doc_id
                    document = dataset.doc_id2document[doc_id]
                    # TODO: get word-level reading predictions
                    readings = [self.id2reading[pred] for pred in reading_preds]
                    word_reading_preds = get_word_level_readings(readings, tokens.split(" "), reading_subword_map)
                    morphemes = self._create_morphemes(
                        [m.text for m in document.morphemes],
                        word_reading_preds,
                        pos_preds,
                        subpos_preds,
                        conjtype_preds,
                        conjform_preds,
                    )
                    document = self._chunk_morphemes(document, morphemes, word_feature_preds)
                    document.doc_id = doc_id
                    self._add_base_phrase_features(document, base_phrase_feature_preds)
                    self._add_named_entities(document, ne_tag_preds)
                    self._add_dependency(document, dependency_preds, dependency_type_preds, dataset.special_to_index)
                    self._add_cohesion(
                        document,
                        cohesion_preds,
                        dataset.cohesion_task_to_rel_types,
                        dataset.exophora_referents,
                        dataset.index_to_special,
                        dataset.extractors,
                    )
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
        reading_preds: list[str],
        pos_preds: list[int],
        subpos_preds: list[int],
        conjtype_preds: list[int],
        conjform_preds: list[int],
    ) -> list[Morpheme]:
        morphemes = []
        for word, reading, pos_index, subpos_index, conjtype_index, conjform_index in zip(
            words, reading_preds, pos_preds, subpos_preds, conjtype_preds, conjform_preds
        ):
            pos = INDEX2POS_TYPE[pos_index]
            pos_id = POS_TYPE2ID[pos]
            subpos = INDEX2SUBPOS_TYPE[subpos_index]
            subpos_id = POS_SUBPOS_TYPE2ID[pos][subpos]
            conjtype = INDEX2CONJTYPE_TYPE[conjtype_index]
            conjtype_id = conjtype_index
            conjform = INDEX2CONJFORM_TYPE[conjform_index]
            conjform_id = CONJTYPE_CONJFORM_TYPE2ID[conjtype][conjform]
            semantics: dict[str, Union[str, bool]] = {}
            homograph_ops: list[dict[str, Any]] = []

            # create lemma using surf, conjtype and conjform
            # TODO: replace word with norm
            if conjtype == "*":
                lemma = word
            else:
                for ambig_surf_spec in self.ambig_surf_specs:
                    if conjtype != ambig_surf_spec["conjtype"]:
                        continue
                    if conjform != ambig_surf_spec["conjform"]:
                        continue
                    # ambiguous: dictionary lookup to identify the lemma
                    q = Query()
                    matches = self.jumandic.search(
                        (q.pos == pos) & (q.subpos == subpos) & (q.conjtype == conjtype) & (q.surf == word)
                    )
                    if len(matches) > 0:
                        lemma_set: set[str] = set()
                        for entry in matches:
                            lemma_set.add(entry["lemma"])
                        lemmas: list[str] = list(lemma_set)
                        lemma = lemmas[0]
                        if len(lemmas) > 1:
                            homograph_ops.append({"type": "lemma", "values": lemmas[1:]})
                    else:
                        logger.warning(f"failed to get lemma for {word}")
                        lemma = word
                    break
                else:
                    # not ambiguous: use paradigm table to generate the lemma
                    try:
                        lemma = self.jinf(word, conjtype, conjform, "基本形")
                    except ValueError as e:
                        logger.warning(f"failed to get lemma for {word}: ({e})")
                        lemma = word
            q = Query()
            matches = self.jumandic.search(
                (q.pos == pos)
                & (q.subpos == subpos)
                & (q.conjtype == conjtype)
                & (q.surf == word)
                & (q.reading == reading)
            )
            if len(matches) > 0:
                entry = matches[0]
                semantics.update(SemanticsDict.from_sstring('"' + entry["semantics"] + '"'))
                if len(matches) > 1:
                    # homograph
                    semantics_list = []
                    for entry2 in matches[1:]:
                        semantics2: dict[str, Union[str, bool]] = {}
                        semantics2.update(SemanticsDict.from_sstring('"' + entry2["semantics"] + '"'))
                        semantics_list.append(semantics2)
                    homograph_ops.append(
                        {
                            "type": "semantics",
                            "values": semantics_list,
                        }
                    )
            attributes = MorphemeAttributes(
                reading=reading,
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
            morpheme = Morpheme(word, attributes, SemanticsDict(semantics), FeatureDict(semantics))
            morphemes.append(morpheme)
            if len(homograph_ops) >= 1:
                range_list = []
                for homograph_op in homograph_ops:
                    range_list.append(range(len(homograph_op["values"])))
                for op_idx_list in itertools.product(*range_list):
                    attributes2 = copy.deepcopy(attributes)
                    semantics2 = copy.deepcopy(semantics)
                    for i, j in enumerate(op_idx_list):
                        homograph_op = homograph_ops[i]
                        v = homograph_op["values"][j]
                        if homograph_op["type"] == "lemma":
                            setattr(attributes2, "lemma", v)
                        elif homograph_op["type"] == "semantics":
                            semantics2 = v
                        else:
                            raise NotImplementedError
                    morpheme2 = Morpheme(
                        word, attributes2, SemanticsDict(semantics2), FeatureDict(semantics2), homograph=True
                    )
                    # rhoknp coverts homographs into KNP's ALT features
                    morpheme.homographs.append(morpheme2)
        return morphemes

    @staticmethod
    def _get_mrph_type_preds(
        pos_logits: torch.Tensor,
        subpos_logits: torch.Tensor,
        conjtype_logits: torch.Tensor,
        conjform_logits: torch.Tensor,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
        batch_pos_preds: list[list[int]] = torch.argmax(pos_logits, dim=-1).tolist()
        batch_subpos_logits: list[list[list[float]]] = subpos_logits.tolist()
        refined_batch_subpos_preds: list[list[int]] = []
        for batch_idx, batch_pos_pred in enumerate(batch_pos_preds):
            refined_subpos_preds: list[int] = []
            for pos_idx, pos in enumerate(batch_pos_pred):
                possible_subpos_ids: set[int] = {
                    SUBPOS_TYPES.index(x) for x in POS_SUBPOS_TYPE2ID[INDEX2POS_TYPE[pos]].keys()
                }
                refined_subpos_pred: int = 0
                refined_subpos_logit: float = float("-inf")
                for logit_idx, logit in enumerate(batch_subpos_logits[batch_idx][pos_idx]):
                    if logit_idx in possible_subpos_ids and logit > refined_subpos_logit:
                        refined_subpos_pred = logit_idx
                        refined_subpos_logit = logit
                refined_subpos_preds.append(refined_subpos_pred)
            refined_batch_subpos_preds.append(refined_subpos_preds)

        batch_conjtype_preds: list[list[int]] = torch.argmax(conjtype_logits, dim=-1).tolist()
        batch_conjform_logits: list[list[list[float]]] = conjform_logits.tolist()
        refined_batch_conjform_preds: list[list[int]] = []
        for batch_idx, batch_conjtype_pred in enumerate(batch_conjtype_preds):
            refined_conjform_preds: list[int] = []
            for conjtype_idx, conjtype in enumerate(batch_conjtype_pred):
                possible_conjform_ids: set[int] = {
                    CONJFORM_TYPES.index(x) for x in CONJTYPE_CONJFORM_TYPE2ID[INDEX2CONJTYPE_TYPE[conjtype]].keys()
                }
                refined_conjform_pred: int = 0
                refined_conjform_logit: float = float("-inf")
                for logit_idx, logit in enumerate(batch_conjform_logits[batch_idx][conjtype_idx]):
                    if logit_idx in possible_conjform_ids and logit > refined_conjform_logit:
                        refined_conjform_pred = logit_idx
                        refined_conjform_logit = logit
                refined_conjform_preds.append(refined_conjform_pred)
            refined_batch_conjform_preds.append(refined_conjform_preds)

        return batch_pos_preds, refined_batch_subpos_preds, batch_conjtype_preds, refined_batch_conjform_preds

    @staticmethod
    def _chunk_morphemes(
        document: Document, morphemes: list[Morpheme], word_feature_preds: list[list[float]]
    ) -> Document:
        assert len(morphemes) <= len(word_feature_preds)
        sentences = []
        for sentence in document.sentences:
            phrases_buff = []
            base_phrases_buff = []
            morphemes_buff = []
            for i in [m.global_index for m in sentence.morphemes]:
                morpheme = morphemes[i]
                morphemes_buff.append(morpheme)
                # follows kwja.utils.constants.WORD_FEATURES
                (
                    base_phrase_head_prob,
                    base_phrase_end_prob,
                    phrase_end_prob,
                    declinable_word_surf_head,
                    declinable_word_surf_end,
                ) = word_feature_preds[i]
                if base_phrase_head_prob >= 0.5:
                    morpheme.features["基本句-主辞"] = True
                # TODO: refactor & set strict condition
                if declinable_word_surf_head >= 0.5:
                    morpheme.features["用言表記先頭"] = True
                if declinable_word_surf_end >= 0.5:
                    morpheme.features["用言表記末尾"] = True
                # even if base_phrase_end_prob is low, if phrase_end_prob is high enough, create chunk here
                if base_phrase_end_prob >= 0.5 or base_phrase_end_prob + phrase_end_prob >= 1.0:
                    base_phrase = BasePhrase(parent_index=None, dep_type=None)
                    base_phrase.morphemes = morphemes_buff
                    morphemes_buff = []
                    base_phrases_buff.append(base_phrase)
                # even if phrase_end_prob is high, if base_phrase_end_prob is not high enough, do not create chunk here
                if phrase_end_prob >= 0.5 and base_phrase_end_prob + phrase_end_prob >= 1.0:
                    phrase = Phrase(parent_index=None, dep_type=None)
                    phrase.base_phrases = base_phrases_buff
                    base_phrases_buff = []
                    phrases_buff.append(phrase)

            # clear buffers
            if morphemes_buff:
                base_phrase = BasePhrase(parent_index=None, dep_type=None)
                base_phrase.morphemes = morphemes_buff
                base_phrases_buff.append(base_phrase)
            if base_phrases_buff:
                phrase = Phrase(parent_index=None, dep_type=None)
                phrase.base_phrases = base_phrases_buff
                phrases_buff.append(phrase)

            sentence = sentence.reparse()
            sentence.phrases = phrases_buff
            sentences.append(sentence)
        return Document.from_sentences(sentences)

    @staticmethod
    def _add_base_phrase_features(document: Document, base_phrase_feature_preds: list[list[float]]) -> None:
        base_phrases = document.base_phrases
        clause_start_index_set: set[int] = {0}
        for base_phrase in base_phrases:
            for feature, pred in zip(BASE_PHRASE_FEATURES, base_phrase_feature_preds[base_phrase.head.global_index]):
                if feature != "節-主辞" and pred >= 0.5:
                    k, *vs = feature.split(":")
                    base_phrase.features[k] = ":".join(vs) or True
                    if feature.startswith("節-区切"):
                        clause_start_index_set.add(base_phrase.global_index + 1)
        if base_phrases[-1].features.get("節-区切", False) is False:
            clause_start_index_set.add(len(base_phrases))
        clause_start_indices: list[int] = sorted(clause_start_index_set)

        for span in zip(clause_start_indices[:-1], clause_start_indices[1:]):
            clause = base_phrases[slice(*span)]
            clause_head_scores = [
                base_phrase_feature_preds[base_phrase.head.global_index][BASE_PHRASE_FEATURES.index("節-主辞")]
                for base_phrase in clause
            ]
            clause_head = clause[clause_head_scores.index(max(clause_head_scores))]
            clause_head.features["節-主辞"] = True

    @staticmethod
    def _add_named_entities(document: Document, ne_tag_preds: list[int]) -> None:
        for sentence in document.sentences:
            morphemes = sentence.morphemes
            category = ""
            morphemes_buff = []
            for morpheme, ne_tag_pred in zip(morphemes, ne_tag_preds):
                ne_tag: str = NE_TAGS[ne_tag_pred]
                if ne_tag.startswith("B-"):
                    category = ne_tag[2:]
                    morphemes_buff.append(morpheme)
                elif ne_tag.startswith("I-") and ne_tag[2:] == category:
                    morphemes_buff.append(morpheme)
                else:
                    if morphemes_buff:
                        named_entity = NamedEntity(category=NamedEntityCategory(category), morphemes=morphemes_buff)
                        # NE feature must be tagged to the last base phrase the named entity contains
                        morphemes_buff[-1].base_phrase.features[
                            "NE"
                        ] = f"{named_entity.category.value}:{named_entity.text}"
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
                return

        for dst in range(src - 1, -1, -1):
            dependency_manager.add_edge(src, dst)
            if dependency_manager.has_cycle():
                dependency_manager.remove_edge(src, dst)
            else:
                base_phrase.parent_index = dst
                base_phrase.dep_type = DepType.DEPENDENCY
                return

        raise RuntimeError("couldn't resolve dependency")

    def _add_dependency(
        self,
        document: Document,
        dependency_preds: list[list[int]],
        dependency_type_preds: list[list[int]],
        special_to_index: dict[str, int],
    ) -> None:
        for sentence in extract_target_sentences(document):
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
        task_to_rel_types: dict[CohesionTask, list[str]],
        exophora_referents: list[ExophoraReferent],
        index_to_special: dict[int, str],
        task_to_extractors: dict[CohesionTask, Extractor],
    ) -> None:
        all_rel_types = [t for ts in task_to_rel_types.values() for t in ts]
        for base_phrase in document.base_phrases:
            base_phrase.rels = RelTagList()
            rel_tags = self._to_rels(
                [preds[base_phrase.head.global_index] for preds in cohesion_preds],
                document.morphemes,
                all_rel_types,
                exophora_referents,
                index_to_special,
            )
            for task, rel_types in task_to_rel_types.items():
                extractor = task_to_extractors[task]
                if extractor.is_target(base_phrase):
                    base_phrase.rels += [rel for rel in rel_tags if rel.type in rel_types]

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
        rel_tags = RelTagList()
        assert len(rel_types) == len(prediction)
        for relation, morpheme_index in zip(rel_types, prediction):
            if morpheme_index < 0:
                continue  # non-target phrase
            if 0 <= morpheme_index < len(morphemes):
                # endophora
                prediction_bp: BasePhrase = morphemes[morpheme_index].base_phrase
                rel_tags.append(
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
                    rel_tags.append(
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

        return rel_tags

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
