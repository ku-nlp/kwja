from logging import getLogger
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from cohesion_tools.extractors.base import BaseExtractor
from rhoknp import BasePhrase, Document, Morpheme, Phrase, Sentence
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType, RelTag
from rhoknp.props import DepType, NamedEntity, NamedEntityCategory

from kwja.datamodule.examples import SpecialTokenIndexer
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
    SENT_SEGMENTATION_TAGS,
    SUBPOS_TAGS,
    TOKEN2TYPO_CORR_OP_TAG,
    WORD_FEATURES,
    WORD_NORM_OP_TAGS,
    WORD_SEGMENTATION_TAGS,
    CohesionTask,
)
from kwja.utils.dependency_parsing import DependencyManager
from kwja.utils.word_normalization import get_normalized

logger = getLogger(__name__)


# ---------- typo module writer ----------
def convert_typo_predictions_into_tags(
    predictions: List[int],  # kdr_predictions or ins_predictions
    probabilities: List[float],
    prefix: Literal["R", "I"],
    confidence_threshold: float,
    token2token_id: Dict[str, int],
    token_id2token: Dict[int, str],
) -> List[str]:
    typo_corr_op_tags: List[str] = []
    for token_id, probability in zip(predictions, probabilities):
        # do not edit if the probability (replace, delete, and insert) is less than "confidence_threshold"
        if probability < confidence_threshold:
            token_id = token2token_id["<k>"] if prefix == "R" else token2token_id["<_>"]
        token: str = token_id2token[token_id]
        typo_corr_op_tag = TOKEN2TYPO_CORR_OP_TAG.get(token, f"{prefix}:{token}")
        typo_corr_op_tags.append(typo_corr_op_tag)
    return typo_corr_op_tags


def apply_edit_operations(pre_text: str, kdr_tags: List[str], ins_tags: List[str]) -> str:
    assert len(pre_text) + 1 == len(kdr_tags) + 1 == len(ins_tags)
    post_text = ""
    for i, char in enumerate(pre_text):
        if ins_tags[i].startswith("I:"):
            post_text += ins_tags[i][2:]  # remove prefix "I:"

        if kdr_tags[i] in {"K", "_"}:  # "_"はinsert用のタグなのでkdr_tags[i] == "_"となることは滅多にない
            post_text += char
        elif kdr_tags[i] == "D":
            pass
        elif kdr_tags[i].startswith("R:"):
            post_text += kdr_tags[i][2:]  # remove prefix "R:"
    if ins_tags[-1].startswith("I:"):
        post_text += ins_tags[-1][2:]  # remove prefix "I:"
    return post_text


# ---------- char module writer ----------
def convert_char_predictions_into_tags(
    sent_segmentation_predictions: List[int],
    word_segmentation_predictions: List[int],
    word_norm_op_predictions: List[int],
    indices: List[int],
) -> Tuple[List[str], List[str], List[str]]:
    sent_segmentation_tags = [SENT_SEGMENTATION_TAGS[sent_segmentation_predictions[i]] for i in indices]
    word_segmentation_tags = [WORD_SEGMENTATION_TAGS[word_segmentation_predictions[i]] for i in indices]
    word_norm_op_tags = [WORD_NORM_OP_TAGS[word_norm_op_predictions[i]] for i in indices]
    return sent_segmentation_tags, word_segmentation_tags, word_norm_op_tags


def set_sentences(document: Document, sent_segmentation_tags: List[str]) -> None:
    sentences: List[Sentence] = []
    surf: str = ""
    for char, sent_segmentation_tag in zip(document.text, sent_segmentation_tags):
        if sent_segmentation_tag == "B" and surf:
            sentences.append(Sentence(surf))
            surf = ""
        surf += char
    if surf:
        sentences.append(Sentence(surf))
    document.sentences = sentences


def set_morphemes(document: Document, word_segmentation_tags: List[str], word_norm_op_tags: List[str]) -> None:
    char_index = 0
    for sentence in document.sentences:
        Morpheme.count = 0
        morphemes: List[Morpheme] = []
        surf: str = ""
        ops: List[str] = []
        for char in sentence.text:
            if word_segmentation_tags[char_index] == "B" and surf:
                norm = get_normalized(surf, ops, strict=False)
                morphemes.append(_build_morpheme(surf, norm))
                surf = ""
                ops = []
            surf += char
            ops.append(word_norm_op_tags[char_index])
            char_index += 1
        if surf:
            norm = get_normalized(surf, ops, strict=False)
            morphemes.append(_build_morpheme(surf, norm))
        sentence.morphemes = morphemes


def _build_morpheme(surf: str, norm: str) -> Morpheme:
    return Morpheme(
        surf,
        reading="_",
        lemma=norm or surf,
        pos="未定義語",
        pos_id=15,
        subpos="その他",
        subpos_id=1,
        conjtype="*",
        conjtype_id=0,
        conjform="*",
        conjform_id=0,
    )


# ---------- word module writer ----------
def get_morpheme_attribute_predictions(
    pos_logits: List[List[float]],
    subpos_logits: List[List[float]],
    conjtype_logits: List[List[float]],
    conjform_logits: List[List[float]],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    pos_predictions: List[int] = np.array(pos_logits).argmax(axis=1).tolist()
    subpos_predictions: List[int] = []
    conjtype_predictions: List[int] = np.array(conjtype_logits).argmax(axis=1).tolist()
    conjform_predictions: List[int] = []
    for i, (pos_index, subpos_logit_list, conjtype_index, conjform_logit_list) in enumerate(
        zip(pos_predictions, subpos_logits, conjtype_predictions, conjform_logits)
    ):
        pos_tag = POS_TAGS[pos_index]
        possible_subpos_tags: Set[str] = set(POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos_tag].keys())
        for subpos_index, subpos_tag in enumerate(SUBPOS_TAGS):
            if subpos_tag not in possible_subpos_tags:
                subpos_logit_list[subpos_index] = MASKED
        subpos_index = np.array(subpos_logit_list).argmax().item()
        subpos_tag = SUBPOS_TAGS[subpos_index]
        assert subpos_tag in possible_subpos_tags
        subpos_predictions.append(subpos_index)

        conjtype_tag = CONJTYPE_TAGS[conjtype_index]
        if (pos_tag, subpos_tag) in INFLECTABLE:
            possible_conjform_tags: Set[str] = set(CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID[conjtype_tag].keys())
            for conjform_index, conjform_tag in enumerate(CONJFORM_TAGS):
                if conjform_tag not in possible_conjform_tags:
                    conjform_logit_list[conjform_index] = MASKED
            conjform_index = np.array(conjform_logit_list).argmax().item()
            conjform_tag = CONJFORM_TAGS[conjform_index]
            assert conjform_tag in possible_conjform_tags
        else:
            conjtype_predictions[i] = CONJTYPE_TAGS.index("*")
            conjform_index = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID["*"]["*"]
        conjform_predictions.append(conjform_index)

    return pos_predictions, subpos_predictions, conjtype_predictions, conjform_predictions


def build_morphemes(
    surfs: List[str],
    lemmas: List[str],
    reading_predictions: List[str],
    morpheme_attribute_predictions: Tuple[List[int], List[int], List[int], List[int]],
) -> List[Morpheme]:
    morphemes = []
    for surf, lemma, reading, pos_index, subpos_index, conjtype_index, conjform_index in zip(
        surfs, lemmas, reading_predictions, *morpheme_attribute_predictions
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
        morpheme_buffer: List[Morpheme] = []
        base_phrase_buffer: List[BasePhrase] = []
        phrase_buffer: List[Phrase] = []
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
    special_token_indexer: SpecialTokenIndexer,
) -> None:
    base_phrases = sentence.base_phrases
    morpheme_global_index2base_phrase_index = {m.global_index: bp.index for bp in base_phrases for m in bp.morphemes}
    morpheme_global_index2base_phrase_index[special_token_indexer.get_morpheme_level_index("[ROOT]")] = -1
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

    raise RuntimeError("couldn't resolve dependency")  # pragma: no cover


def add_cohesion(
    document: Document,
    cohesion_logits: List[List[List[float]]],  # (rel, seq, seq)
    cohesion_task2extractor: Dict[CohesionTask, BaseExtractor],
    cohesion_task2rels: Dict[CohesionTask, List[str]],
    restrict_cohesion_target: bool,
    special_token_indexer: SpecialTokenIndexer,
) -> None:
    rel2logits = dict(
        zip(
            [r for cohesion_rels in cohesion_task2rels.values() for r in cohesion_rels],
            cohesion_logits,
        )
    )
    base_phrases = document.base_phrases
    for base_phrase in base_phrases:
        base_phrase.rel_tags.clear()
        for cohesion_task, cohesion_extractor in cohesion_task2extractor.items():
            if restrict_cohesion_target is True and cohesion_extractor.is_target(base_phrase) is False:
                continue
            for rel in cohesion_task2rels[cohesion_task]:
                rel_tag = _to_rel_tag(
                    rel,
                    rel2logits[rel][base_phrase.head.global_index],  # (seq, )
                    base_phrases,
                    special_token_indexer,
                    cohesion_extractor.exophora_referent_types,
                )
                if rel_tag is not None:
                    base_phrase.rel_tags.append(rel_tag)


def _to_rel_tag(
    rel: str,
    rel_logits: List[float],  # (seq, )
    base_phrases: List[BasePhrase],
    special_token_indexer: SpecialTokenIndexer,
    exophora_referent_types: List[ExophoraReferentType],
) -> Optional[RelTag]:
    logits = [rel_logits[bp.head.global_index] for bp in base_phrases]
    logits += [rel_logits[i] for i in special_token_indexer.get_morpheme_level_indices()]
    predicted_base_phrase_global_index: int = np.argmax(logits).item()
    if 0 <= predicted_base_phrase_global_index < len(base_phrases):
        # endophora
        predicted_antecedent = base_phrases[predicted_base_phrase_global_index]
        return RelTag(
            type=rel,
            target=predicted_antecedent.head.text,
            sid=predicted_antecedent.sentence.sid,
            base_phrase_index=predicted_antecedent.index,
            mode=None,
        )
    else:
        # exophora
        special_token = special_token_indexer.special_tokens[predicted_base_phrase_global_index - len(base_phrases)]
        stripped_special_token = special_token[1:-1]  # strip '[' and ']'
        if ExophoraReferent(stripped_special_token).type in exophora_referent_types:  # exclude [NULL], [NA], and [ROOT]
            return RelTag(
                type=rel,
                target=stripped_special_token,
                sid=None,
                base_phrase_index=None,
                mode=None,
            )
    return None


def add_discourse(document: Document, discourse_predictions: List[List[int]]) -> None:
    if document.is_clause_tag_required():
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
