import logging
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
from jinf import Jinf
from rhoknp import Document, Morpheme, Sentence
from rhoknp.props import SemanticsDict

from kwja.callbacks.base_module_writer import BaseModuleWriter
from kwja.callbacks.utils import (
    add_base_phrase_features,
    add_cohesion,
    add_dependency,
    add_discourse,
    add_named_entities,
    chunk_morphemes,
    get_morpheme_attribute_predictions,
)
from kwja.datamodule.datasets import WordDataset, WordInferenceDataset
from kwja.datamodule.examples import WordExample, WordInferenceExample
from kwja.utils.constants import (
    CONJFORM_TAGS,
    CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID,
    CONJTYPE_TAGS,
    POS_TAG2POS_ID,
    POS_TAG_SUBPOS_TAG2SUBPOS_ID,
    POS_TAGS,
    RESOURCE_PATH,
    SUBPOS_TAGS,
)
from kwja.utils.jumandic import JumanDic
from kwja.utils.reading_prediction import get_reading2reading_id, get_word_level_readings
from kwja.utils.sub_document import extract_target_sentences, to_orig_doc_id

logger = logging.getLogger(__name__)


class WordModuleWriter(BaseModuleWriter):
    def __init__(
        self,
        ambig_surf_specs: List[Dict[str, str]],
        preserve_reading_lemma_canon: bool = False,
        destination: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(destination=destination)
        reading2reading_id = get_reading2reading_id(RESOURCE_PATH / "reading_prediction" / "vocab.txt")
        self.reading_id2reading = {v: k for k, v in reading2reading_id.items()}

        self.ambig_surf_specs = ambig_surf_specs
        self.jumandic = JumanDic(RESOURCE_PATH / "jumandic")
        self.jinf = Jinf()

        self.preserve_reading_lemma_canon: bool = preserve_reading_lemma_canon

        self.doc_id_sid2predicted_sentence: Dict[str, Dict[str, Sentence]] = defaultdict(dict)
        self.prev_doc_id = ""

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
        if isinstance(trainer.predict_dataloaders, dict):
            dataloader = list(trainer.predict_dataloaders.values())[dataloader_idx]
        else:
            dataloader = trainer.predict_dataloaders[dataloader_idx]
        dataset: Union[WordDataset, WordInferenceDataset] = dataloader.dataset
        for (
            example_id,
            reading_predictions,
            reading_subword_map,
            *morpheme_attribute_logits,  # pos_logits, subpos_logits, conjtype_logits, conjform_logits
            word_feature_probabilities,
            ne_predictions,
            base_phrase_feature_probabilities,
            dependency_predictions,
            dependency_type_predictions,
            cohesion_logits,
            discourse_predictions,
        ) in zip(*[v.tolist() for v in prediction.values()]):
            example: Union[WordExample, WordInferenceExample] = dataset.examples[example_id]
            assert example.doc_id is not None, "doc_id isn't set"
            document = dataset.doc_id2document.pop(example.doc_id)

            if self.preserve_reading_lemma_canon is True:
                word_reading_predictions = [m.reading for m in document.morphemes]
                canons = [m.canon for m in document.morphemes]
            else:
                word_reading_predictions = get_word_level_readings(
                    [self.reading_id2reading[reading_id] for reading_id in reading_predictions],
                    [dataset.tokenizer.decode(input_id) for input_id in example.encoding.ids],
                    reading_subword_map,
                )
                canons = [None for _ in document.morphemes]
            morpheme_attribute_predictions = get_morpheme_attribute_predictions(
                *(logits[: len(document.morphemes)] for logits in morpheme_attribute_logits)
            )
            morphemes = self._build_morphemes(
                [m.surf for m in document.morphemes],
                [m.lemma for m in document.morphemes],
                word_reading_predictions,
                morpheme_attribute_predictions,
                canons,
                self.preserve_reading_lemma_canon,
            )
            predicted_document = chunk_morphemes(document, morphemes, word_feature_probabilities)
            predicted_document.doc_id = document.doc_id
            for sentence in extract_target_sentences(predicted_document):
                add_named_entities(sentence, ne_predictions)
                add_base_phrase_features(sentence, base_phrase_feature_probabilities)
                add_dependency(
                    sentence, dependency_predictions, dependency_type_predictions, example.special_token_indexer
                )
            orig_doc_id = to_orig_doc_id(document.doc_id)
            # 解析済みの文とマージ
            sentences: List[Sentence] = [
                self.doc_id_sid2predicted_sentence[orig_doc_id].get(s.sid, s) for s in predicted_document.sentences
            ]
            predicted_document = Document.from_sentences(sentences)
            predicted_document.doc_id = document.doc_id
            add_cohesion(
                predicted_document,
                cohesion_logits,
                dataset.cohesion_task2extractor,
                dataset.cohesion_task2rels,
                dataset.restrict_cohesion_target,
                example.special_token_indexer,
            )
            add_discourse(predicted_document, discourse_predictions)

            for predicted_sentence, sentence in zip(
                extract_target_sentences(predicted_document),
                extract_target_sentences(document),
            ):
                predicted_sentence.comment = sentence.comment
                self.doc_id_sid2predicted_sentence[orig_doc_id][predicted_sentence.sid] = predicted_sentence

            if orig_doc_id != self.prev_doc_id:
                sid2predicted_sentence = self.doc_id_sid2predicted_sentence[self.prev_doc_id]
                output_string = "".join(s.to_knp() for s in sid2predicted_sentence.values())
                self.write_output_string(output_string)
                self.doc_id_sid2predicted_sentence[self.prev_doc_id].clear()
                self.prev_doc_id = orig_doc_id

        if batch_idx == len(dataloader) - 1:
            for sid2predicted_sentence in self.doc_id_sid2predicted_sentence.values():
                output_string = "".join(s.to_knp() for s in sid2predicted_sentence.values())
                self.write_output_string(output_string)
            self.doc_id_sid2predicted_sentence.clear()

    def _build_morphemes(
        self,
        surfs: List[str],
        norms: List[str],
        reading_predictions: List[str],
        morpheme_attribute_predictions: Tuple[List[int], List[int], List[int], List[int]],
        canons: List[Optional[str]],
        preserve_lemma: bool,
    ) -> List[Morpheme]:
        assert len(surfs) == len(norms) == len(reading_predictions)
        morphemes = []
        for surf, norm, reading, pos_index, subpos_index, conjtype_index, conjform_index, canon in zip(
            surfs, norms, reading_predictions, *morpheme_attribute_predictions, canons
        ):
            pos = POS_TAGS[pos_index]
            pos_id = POS_TAG2POS_ID[pos]
            subpos = SUBPOS_TAGS[subpos_index]
            subpos_id = POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos][subpos]
            conjtype = CONJTYPE_TAGS[conjtype_index]
            conjtype_id = conjtype_index
            conjform = CONJFORM_TAGS[conjform_index]
            conjform_id = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID[conjtype][conjform]

            homograph_ops: List[Dict[str, Any]] = []
            if preserve_lemma is True:
                lemma = norm
            else:
                lemma = self._get_lemma(norm, pos, subpos, conjtype, conjform, homograph_ops)
            semantics = self._lookup_semantics(reading, norm, pos, subpos, conjtype, canon, homograph_ops)
            morpheme = Morpheme(
                surf,
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
                semantics=semantics,
            )
            if len(homograph_ops) >= 1:
                self._add_homographs(homograph_ops, morpheme, semantics)
            morphemes.append(morpheme)
        return morphemes

    def _get_lemma(
        self, norm: str, pos: str, subpos: str, conjtype: str, conjform: str, homograph_ops: List[Dict[str, Any]]
    ) -> str:
        # get lemma using norm, pos, subpos, conjtype and conjform
        if conjtype == "*":
            lemma: str = norm
        else:
            for ambig_surf_spec in self.ambig_surf_specs:
                if conjtype != ambig_surf_spec["conjtype"] or conjform != ambig_surf_spec["conjform"]:
                    continue
                # ambiguous: dictionary lookup to identify the lemma
                matched = [
                    entry
                    for entry in self.jumandic.lookup_by_norm(norm)
                    if entry["pos"] == pos and entry["subpos"] == subpos and entry["conjtype"] == conjtype
                ]
                if len(matched) > 0:
                    lemmas: List[str] = [*{entry["lemma"] for entry in matched}]
                    lemma = lemmas[0]
                    if len(lemmas) > 1:
                        homograph_ops.append({"type": "lemma", "values": lemmas[1:]})
                else:
                    logger.warning(f"failed to get lemma for {norm}")
                    lemma = norm
                break
            else:
                # not ambiguous: use paradigm table to generate the lemma
                try:
                    lemma = self.jinf(norm, conjtype, conjform, "基本形")
                except ValueError as e:
                    logger.warning(f"failed to get lemma for {norm}: ({e})")
                    lemma = norm
        return lemma

    def _lookup_semantics(
        self,
        reading: str,
        norm: str,
        pos: str,
        subpos: str,
        conjtype: str,
        canon: Optional[str],
        homograph_ops: List[Dict[str, Any]],
    ) -> SemanticsDict:
        if canon is not None:
            matched = [
                entry
                for entry in self.jumandic.lookup_by_canon(canon)
                if (
                    entry["reading"] == reading
                    and entry["lemma"] == norm
                    and entry["pos"] == pos
                    and entry["subpos"] == subpos
                    and entry["conjtype"] == conjtype
                )
            ]
        else:
            matched = [
                entry
                for entry in self.jumandic.lookup_by_norm(norm)
                if (
                    entry["reading"] == reading
                    and entry["pos"] == pos
                    and entry["subpos"] == subpos
                    and entry["conjtype"] == conjtype
                )
            ]
        if len(matched) > 0:
            entry = matched[0]
            semantics: SemanticsDict = SemanticsDict.from_sstring('"' + entry["semantics"] + '"')
            if len(matched) > 1:
                # homograph
                homograph_ops.append(
                    {
                        "type": "semantics",
                        "values": [
                            SemanticsDict.from_sstring('"' + homograph_entry["semantics"] + '"')
                            for homograph_entry in matched[1:]
                        ],
                    }
                )
        else:
            semantics = SemanticsDict()
        return semantics

    @staticmethod
    def _add_homographs(homograph_ops: List[Dict[str, Any]], morpheme: Morpheme, semantics: SemanticsDict) -> None:
        range_list = [range(len(homograph_op["values"])) for homograph_op in homograph_ops]
        for op_idx_list in product(*range_list):
            homograph_lemma = morpheme.lemma
            homograph_semantics = SemanticsDict.from_sstring(semantics.to_sstring())
            for i, j in enumerate(op_idx_list):
                homograph_op = homograph_ops[i]
                v = homograph_op["values"][j]
                if homograph_op["type"] == "lemma":
                    homograph_lemma = v
                elif homograph_op["type"] == "semantics":
                    homograph_semantics = v
                else:
                    raise NotImplementedError
            homograph = Morpheme(
                morpheme.surf,
                reading=morpheme.reading,
                lemma=homograph_lemma,
                pos=morpheme.pos,
                pos_id=morpheme.pos_id,
                subpos=morpheme.subpos,
                subpos_id=morpheme.subpos_id,
                conjtype=morpheme.conjtype,
                conjtype_id=morpheme.conjtype_id,
                conjform=morpheme.conjform,
                conjform_id=morpheme.conjform_id,
                semantics=homograph_semantics,
                homograph=True,
            )
            # rhoknp converts homographs into KNP's ALT features
            morpheme.homographs.append(homograph)
