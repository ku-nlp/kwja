from logging import getLogger
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from rhoknp import BasePhrase, Document, Morpheme, Phrase, Sentence
from rhoknp.cohesion import ExophoraReferent, RelTag, RelTagList
from rhoknp.props import DepType, NamedEntity, NamedEntityCategory
from transformers import PreTrainedTokenizerBase

from kwja.utils.cohesion_analysis import CohesionUtils
from kwja.utils.constants import (
    BASE_PHRASE_FEATURES,
    CONJFORM_TAGS,
    CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID,
    CONJTYPE_TAGS,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    INFLECTABLE,
    MASKED,
    NE_TAGS,
    POS_TAG2POS_ID,
    POS_TAG_SUBPOS_TAG2SUBPOS_ID,
    POS_TAGS,
    SUBPOS_TAGS,
    WORD_FEATURES,
    CohesionTask,
)
from kwja.utils.dependency_parsing import DependencyManager
from kwja.utils.reading_prediction import get_word_level_readings

logger = getLogger(__name__)


def get_word_reading_predictions(
    input_ids: torch.Tensor,
    reading_predictions: List[int],
    reading_id2reading: Dict[int, str],
    tokenizer: PreTrainedTokenizerBase,
    reading_subword_map: List[List[bool]],
) -> List[str]:
    readings: List[str] = [reading_id2reading[reading_id] for reading_id in reading_predictions]
    tokens: List[str] = [tokenizer.decode(input_id) for input_id in input_ids]
    word_reading_predictions: List[str] = get_word_level_readings(readings, tokens, reading_subword_map)
    return word_reading_predictions


def get_morpheme_attribute_predictions(
    pos_logits: torch.Tensor,
    subpos_logits: torch.Tensor,
    conjtype_logits: torch.Tensor,
    conjform_logits: torch.Tensor,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    pos_predictions: List[int] = pos_logits.argmax(dim=1).tolist()
    subpos_predictions: List[int] = []
    for pos_index, subpos_logit_list in zip(pos_predictions, subpos_logits.tolist()):
        subpos_tag2subpos_id = POS_TAG_SUBPOS_TAG2SUBPOS_ID[POS_TAGS[pos_index]]
        possible_subpos_indices: Set[int] = {
            SUBPOS_TAGS.index(subpos_tag) for subpos_tag in subpos_tag2subpos_id.keys()
        }
        max_subpos_logit = MASKED
        subpos_prediction: int = 0
        for subpos_index, subpos_logit in enumerate(subpos_logit_list):
            if subpos_index in possible_subpos_indices and subpos_logit > max_subpos_logit:
                max_subpos_logit = subpos_logit
                subpos_prediction = subpos_index
        subpos_predictions.append(subpos_prediction)

    conjtype_predictions: List[int] = conjtype_logits.argmax(dim=1).tolist()
    conjform_predictions: List[int] = []
    for i, (pos_index, subpos_index, conjtype_index, conjform_logit_list) in enumerate(
        zip(pos_predictions, subpos_predictions, conjtype_predictions, conjform_logits.tolist())
    ):
        pos_tag: str = POS_TAGS[pos_index]
        subpos_tag: str = SUBPOS_TAGS[subpos_index]
        if (pos_tag, subpos_tag) in INFLECTABLE:
            conjform_tag2conjform_id = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID[CONJTYPE_TAGS[conjtype_index]]
            possible_conjform_indices: Set[int] = {
                CONJFORM_TAGS.index(conjform_tag) for conjform_tag in conjform_tag2conjform_id.keys()
            }
            max_conjform_logit = MASKED
            conjform_prediction: int = 0
            for conjform_index, conjform_logit in enumerate(conjform_logit_list):
                if conjform_index in possible_conjform_indices and conjform_logit > max_conjform_logit:
                    max_conjform_logit = conjform_logit
                    conjform_prediction = conjform_index
        else:
            conjtype_predictions[i] = CONJTYPE_TAGS.index("*")
            conjform_prediction = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID["*"]["*"]
        conjform_predictions.append(conjform_prediction)

    return pos_predictions, subpos_predictions, conjtype_predictions, conjform_predictions


def build_morphemes(
    surfs: List[str],
    lemmas: List[str],
    reading_predictions: List[str],
    pos_predictions: List[int],
    subpos_predictions: List[int],
    conjtype_predictions: List[int],
    conjform_predictions: List[int],
) -> List[Morpheme]:
    morphemes = []
    for surf, lemma, reading, pos_index, subpos_index, conjtype_index, conjform_index in zip(
        surfs,
        lemmas,
        reading_predictions,
        pos_predictions,
        subpos_predictions,
        conjtype_predictions,
        conjform_predictions,
    ):
        pos = POS_TAGS[pos_index]
        pos_id = POS_TAG2POS_ID[pos]
        subpos = SUBPOS_TAGS[subpos_index]
        subpos_id = POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos][subpos]
        conjtype = CONJTYPE_TAGS[conjtype_index]
        conjtype_id = conjtype_index
        conjform = CONJFORM_TAGS[conjform_index]
        conjform_id = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID[conjtype][conjform]
        morphemes.append(
            Morpheme(
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
            )
        )
    return morphemes


def chunk_morphemes(
    document: Document, morphemes: List[Morpheme], word_feature_probabilities: List[List[float]]
) -> Document:
    predicted_sentences = []
    for sentence in document.sentences:
        morpheme_buffer = []
        base_phrase_buffer = []
        phrase_buffer = []
        for i in [m.global_index for m in sentence.morphemes]:
            morpheme = morphemes[i]

            base_phrase_head_probability = word_feature_probabilities[i][WORD_FEATURES.index("基本句-主辞")]
            inflectable_word_surf_head_probability = word_feature_probabilities[i][WORD_FEATURES.index("用言表記先頭")]
            inflectable_word_surf_end_probability = word_feature_probabilities[i][WORD_FEATURES.index("用言表記末尾")]
            if base_phrase_head_probability >= 0.5:
                morpheme.features["基本句-主辞"] = True
            if inflectable_word_surf_head_probability >= 0.5:
                morpheme.features["用言表記先頭"] = True
            if inflectable_word_surf_end_probability >= 0.5:
                morpheme.features["用言表記末尾"] = True
            morpheme_buffer.append(morpheme)

            base_phrase_end_probability = word_feature_probabilities[i][WORD_FEATURES.index("基本句-区切")]
            phrase_end_probability = word_feature_probabilities[i][WORD_FEATURES.index("文節-区切")]
            # even if base_phrase_end_prob is low, if phrase_end_prob is high enough, create chunk here
            if base_phrase_end_probability >= 0.5 or base_phrase_end_probability + phrase_end_probability >= 1.0:
                base_phrase = BasePhrase(parent_index=None, dep_type=None)
                base_phrase.morphemes = morpheme_buffer
                morpheme_buffer = []
                base_phrase_buffer.append(base_phrase)
            # even if phrase_end_prob is high, if base_phrase_end_prob is not high enough, do not create chunk here
            if phrase_end_probability >= 0.5 and base_phrase_end_probability + phrase_end_probability >= 1.0:
                phrase = Phrase(parent_index=None, dep_type=None)
                phrase.base_phrases = base_phrase_buffer
                base_phrase_buffer = []
                phrase_buffer.append(phrase)

        # clear buffers
        if morpheme_buffer:
            base_phrase = BasePhrase(parent_index=None, dep_type=None)
            base_phrase.morphemes = morpheme_buffer
            base_phrase_buffer.append(base_phrase)
        if base_phrase_buffer:
            phrase = Phrase(parent_index=None, dep_type=None)
            phrase.base_phrases = base_phrase_buffer
            phrase_buffer.append(phrase)

        predicted_sentence = Sentence()
        predicted_sentence.comment = sentence.comment
        predicted_sentence.phrases = phrase_buffer
        predicted_sentences.append(predicted_sentence)
    return Document.from_sentences(predicted_sentences)


def add_named_entities(sentence: Sentence, ne_predictions: List[int]) -> None:
    category = ""
    morpheme_buffer: List[Morpheme] = []
    for morpheme in sentence.morphemes:
        ne_index = ne_predictions[morpheme.global_index]
        ne_tag: str = NE_TAGS[ne_index]
        if ne_tag.startswith("B-"):
            _clear_morpheme_buffer(morpheme_buffer, category)
            category = ne_tag[2:]
            morpheme_buffer.append(morpheme)
        elif ne_tag.startswith("I-") and ne_tag[2:] == category:
            morpheme_buffer.append(morpheme)
        else:
            _clear_morpheme_buffer(morpheme_buffer, category)
            category = ""
    else:
        _clear_morpheme_buffer(morpheme_buffer, category)


def _clear_morpheme_buffer(morpheme_buffer: List[Morpheme], category: str) -> None:
    if morpheme_buffer:
        named_entity = NamedEntity(category=NamedEntityCategory(category), morphemes=morpheme_buffer)
        morpheme_buffer[-1].base_phrase.features["NE"] = f"{named_entity.category.value}:{named_entity.text}"
    morpheme_buffer.clear()


def add_base_phrase_features(sentence: Sentence, base_phrase_feature_probabilities: List[List[float]]) -> None:
    phrases = sentence.phrases
    clause_boundary_feature, clause_start_index = None, 0
    for phrase in phrases:
        for base_phrase in phrase.base_phrases:
            for base_phrase_feature, base_phrase_probability in zip(
                BASE_PHRASE_FEATURES, base_phrase_feature_probabilities[base_phrase.head.global_index]
            ):
                if base_phrase_feature.startswith("節-区切") and base_phrase_probability >= 0.5:
                    clause_boundary_feature = base_phrase_feature
                elif base_phrase_feature != "節-主辞" and base_phrase_probability >= 0.5:
                    k, *vs = base_phrase_feature.split(":")
                    base_phrase.features[k] = ":".join(vs) or True
        if phrase == phrases[-1] and clause_boundary_feature is None:
            clause_boundary_feature = "節-区切"

        if clause_boundary_feature is not None:
            k, *vs = clause_boundary_feature.split(":")
            phrase.base_phrases[-1].features[k] = ":".join(vs) or True
            base_phrases = [bp for p in phrases[clause_start_index : phrase.index + 1] for bp in p.base_phrases]
            clause_head_probabilities = [
                base_phrase_feature_probabilities[bp.head.global_index][BASE_PHRASE_FEATURES.index("節-主辞")]
                for bp in base_phrases
            ]
            clause_head = base_phrases[clause_head_probabilities.index(max(clause_head_probabilities))]
            clause_head.features["節-主辞"] = True
            clause_boundary_feature, clause_start_index = None, phrase.index + 1


def add_dependency(
    sentence: Sentence,
    dependency_predictions: List[List[int]],
    dependency_type_predictions: List[List[int]],
    special_token2index: Dict[str, int],
) -> None:
    base_phrases = sentence.base_phrases
    morpheme_global_index2base_phrase_index = {m.global_index: bp.index for bp in base_phrases for m in bp.morphemes}
    morpheme_global_index2base_phrase_index[special_token2index["[ROOT]"]] = -1
    dependency_manager = DependencyManager()
    for base_phrase in base_phrases:
        for parent_morpheme_global_index, dependency_type_index in zip(
            dependency_predictions[base_phrase.head.global_index],
            dependency_type_predictions[base_phrase.head.global_index],
        ):
            parent_index = morpheme_global_index2base_phrase_index[parent_morpheme_global_index]
            dependency_manager.add_edge(base_phrase.index, parent_index)
            if dependency_manager.has_cycle() or (parent_index == -1 and dependency_manager.root):
                dependency_manager.remove_edge(base_phrase.index, parent_index)
            else:
                base_phrase.parent_index = parent_index
                base_phrase.dep_type = DEPENDENCY_TYPES[dependency_type_index]
                break
        else:
            if base_phrase == base_phrases[-1] and not dependency_manager.root:
                base_phrase.parent_index = -1
                base_phrase.dep_type = DepType.DEPENDENCY
            else:
                _resolve_dependency(base_phrase, dependency_manager)

        assert base_phrase.parent_index is not None
        if base_phrase.parent_index == -1:
            base_phrase.phrase.parent_index = -1
            base_phrase.phrase.dep_type = DepType.DEPENDENCY
            dependency_manager.root = True
        elif base_phrase.phrase != base_phrases[base_phrase.parent_index].phrase:
            base_phrase.phrase.parent_index = base_phrases[base_phrase.parent_index].phrase.index
            base_phrase.phrase.dep_type = base_phrase.dep_type


def _resolve_dependency(base_phrase: BasePhrase, dependency_manager: DependencyManager) -> None:
    source = base_phrase.index
    num_base_phrases = len(base_phrase.sentence.base_phrases)
    # 日本語は左から右に係るので、まず右方向に係り先を探す
    for target in range(source + 1, num_base_phrases):
        dependency_manager.add_edge(source, target)
        if dependency_manager.has_cycle():
            dependency_manager.remove_edge(source, target)
        else:
            base_phrase.parent_index = target
            base_phrase.dep_type = DepType.DEPENDENCY
            return

    for target in range(source - 1, -1, -1):
        dependency_manager.add_edge(source, target)
        if dependency_manager.has_cycle():
            dependency_manager.remove_edge(source, target)
        else:
            base_phrase.parent_index = target
            base_phrase.dep_type = DepType.DEPENDENCY
            return

    raise RuntimeError("couldn't resolve dependency")


def add_cohesion(
    document: Document,
    cohesion_logits: List[List[List[float]]],  # (rel, src, tgt)
    cohesion_task2utils: Dict[CohesionTask, CohesionUtils],
    index2special_token: Dict[int, str],
) -> None:
    flatten_rels = [r for cohesion_utils in cohesion_task2utils.values() for r in cohesion_utils.rels]
    base_phrases = document.base_phrases
    for base_phrase in base_phrases:
        rel_tags = RelTagList()
        for cohesion_utils in cohesion_task2utils.values():
            if cohesion_utils.is_target(base_phrase):
                for rel in cohesion_utils.rels:
                    rel_tag = _to_rel_tag(
                        rel,
                        cohesion_logits[flatten_rels.index(rel)][base_phrase.head.global_index],  # (tgt, )
                        base_phrases,
                        index2special_token,
                        cohesion_utils.exophora_referents,
                    )
                    if rel_tag is not None:
                        rel_tags.append(rel_tag)
        base_phrase.rel_tags = rel_tags


def _to_rel_tag(
    rel: str,
    rel_logits: List[float],  # (tgt, )
    base_phrases: List[BasePhrase],
    index2special_token: Dict[int, str],
    exophora_referents: List[ExophoraReferent],
) -> Optional[RelTag]:
    logits = [rel_logits[bp.head.global_index] for bp in base_phrases] + [rel_logits[i] for i in index2special_token]
    predicted_antecedent_index: int = np.argmax(logits).item()
    if 0 <= predicted_antecedent_index < len(base_phrases):
        # endophora
        predicted_antecedent = base_phrases[predicted_antecedent_index]
        return RelTag(
            type=rel,
            target=predicted_antecedent.head.text,
            sid=predicted_antecedent.sentence.sid,
            base_phrase_index=predicted_antecedent.index,
            mode=None,
        )
    else:
        # exophora
        special_token = list(index2special_token.values())[predicted_antecedent_index - len(base_phrases)]
        if special_token in [str(e) for e in exophora_referents]:  # exclude [NULL], [NA], and [ROOT]
            return RelTag(
                type=rel,
                target=special_token,
                sid=None,
                base_phrase_index=None,
                mode=None,
            )
        else:
            return None


def add_discourse(document: Document, discourse_predictions: List[List[int]]) -> None:
    if document.need_clause_tag:
        logger.warning("failed to output clause boundaries")
        return

    for modifier in document.clauses:
        modifier_morpheme_global_index = modifier.end.morphemes[0].global_index
        relation_buffer = []
        for head in document.clauses:
            head_morpheme_global_index = head.end.morphemes[0].global_index
            discourse_index = discourse_predictions[modifier_morpheme_global_index][head_morpheme_global_index]
            discourse_relation = DISCOURSE_RELATIONS[discourse_index]
            if discourse_relation != "談話関係なし":
                relation_buffer.append(f"{head.sentence.sid}/{head.end.index}/{discourse_relation}")
        if relation_buffer:
            modifier.end.features["談話関係"] = ";".join(relation_buffer)
