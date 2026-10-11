import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import torch
from cohesion_tools.extractors import PasExtractor
from cohesion_tools.extractors.base import T
from transformers import PreTrainedTokenizerBase

from kwja.datamodule.datasets import WordInferenceDataset
from kwja.utils.constants import CohesionTask
from kwja.utils.sub_document import extract_target_sentences


def test_init(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
    max_seq_length = 256
    document_split_stride = 1
    _ = WordInferenceDataset(word_tokenizer, max_seq_length, document_split_stride, **dataset_kwargs)


def test_len(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
    max_seq_length = 256
    document_split_stride = 1
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer, max_seq_length, document_split_stride, juman_file=Path(juman_file.name), **dataset_kwargs
        )
    finally:
        juman_file_path.unlink(missing_ok=True)
    assert len(dataset) == 1


def test_len_multi_doc(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
    max_seq_length = 256
    document_split_stride = 1
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        雨 _ 雨 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer, max_seq_length, document_split_stride, juman_file=Path(juman_file.name), **dataset_kwargs
        )
    finally:
        juman_file_path.unlink(missing_ok=True)
    assert len(dataset) == 2


def test_getitem(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
    max_seq_length = 256
    document_split_stride = 1
    juman_text = dedent(
        """\
        # S-ID:test-0-0
        今日 _ 今日 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
        だ _ だ 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer,
            max_seq_length,
            document_split_stride,
            juman_file=juman_file_path,
            **dataset_kwargs,
        )
    finally:
        juman_file_path.unlink(missing_ok=True)

    num_cohesion_rels = len([r for rels in dataset.cohesion_task2rels.values() for r in rels])
    for i in range(len(dataset)):
        feature = dataset[i]
        assert feature.example_ids == i
        assert len(feature.input_ids) == max_seq_length
        assert len(feature.attention_mask) == max_seq_length
        assert len(feature.ne_mask) == max_seq_length
        assert np.array(feature.subword_map).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.reading_subword_map).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.dependency_mask).shape == (max_seq_length, max_seq_length)
        assert np.array(feature.cohesion_mask).shape == (num_cohesion_rels, max_seq_length, max_seq_length)


def test_skipped_documents(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
    max_seq_length = 32
    document_split_stride = 1
    juman_text = (
        dedent(
            """\
            # S-ID:test-0-0
            今日 _ 今日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            # S-ID:test-0-1
            """
        )
        + "あ _ あ 未定義語 15 その他 1 * 0 * 0\n" * 40
        + dedent(
            """\
            EOS
            # S-ID:test-0-2
            明日 _ 明日 未定義語 15 その他 1 * 0 * 0
            は _ は 未定義語 15 その他 1 * 0 * 0
            雨 _ 雨 未定義語 15 その他 1 * 0 * 0
            だ _ だ 未定義語 15 その他 1 * 0 * 0
            EOS
            """
        )
    )
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)

    try:
        dataset = WordInferenceDataset(
            word_tokenizer, max_seq_length, document_split_stride, juman_file=juman_file_path, **dataset_kwargs
        )
    finally:
        juman_file_path.unlink(missing_ok=True)
    assert len(dataset) == 2
    assert len(dataset.skipped_documents) == 1
    assert [sentence.sid for sentence in extract_target_sentences(dataset.skipped_documents[0])] == ["test-0-1"]
    assert dataset.skipped_documents[0].text == "あ" * 40


THREE_SENTENCES = dedent(
    """\
    # S-ID:test-0-0
    今日 _ 今日 未定義語 15 その他 1 * 0 * 0
    は _ は 未定義語 15 その他 1 * 0 * 0
    晴れ _ 晴れ 未定義語 15 その他 1 * 0 * 0
    だ _ だ 未定義語 15 その他 1 * 0 * 0
    EOS
    # S-ID:test-0-1
    明日 _ 明日 未定義語 15 その他 1 * 0 * 0
    は _ は 未定義語 15 その他 1 * 0 * 0
    雨 _ 雨 未定義語 15 その他 1 * 0 * 0
    だ _ だ 未定義語 15 その他 1 * 0 * 0
    EOS
    # S-ID:test-0-2
    明後日 _ 明後日 未定義語 15 その他 1 * 0 * 0
    は _ は 未定義語 15 その他 1 * 0 * 0
    雪 _ 雪 未定義語 15 その他 1 * 0 * 0
    だ _ だ 未定義語 15 その他 1 * 0 * 0
    EOS
    """
)


def _load_dataset(
    juman_text: str, word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]
) -> WordInferenceDataset:
    with tempfile.NamedTemporaryFile("wt", delete=False, encoding="utf-8") as juman_file:
        juman_file.write(juman_text)
        juman_file_path = Path(juman_file.name)
    try:
        return WordInferenceDataset(word_tokenizer, 256, 1, juman_file=juman_file_path, **dataset_kwargs)
    finally:
        juman_file_path.unlink(missing_ok=True)


def _cohesion_mask_from_extractors(dataset: WordInferenceDataset, index: int) -> torch.Tensor:
    """Build the cohesion mask by asking each extractor about each morpheme."""
    example = dataset.examples[index]
    morphemes = dataset.doc_id2document[example.doc_id].morphemes
    special_indices = example.special_token_indexer.get_morpheme_level_indices(only_cohesion=True)
    rel_masks: list[torch.Tensor] = []
    for cohesion_task in dataset.cohesion_tasks:
        cohesion_extractor = dataset.cohesion_task2extractor[cohesion_task]
        rel_mask = torch.zeros((dataset.max_seq_length, dataset.max_seq_length), dtype=torch.bool)
        for morpheme in morphemes:
            for candidate in cohesion_extractor.get_candidates(morpheme, morphemes):
                rel_mask[morpheme.global_index, candidate.global_index] = True
            rel_mask[morpheme.global_index, special_indices] = True
        rel_masks.extend(rel_mask for _ in dataset.cohesion_task2rels[cohesion_task])
    return torch.stack(rel_masks)


def test_cohesion_mask(word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]) -> None:
    dataset = _load_dataset(THREE_SENTENCES, word_tokenizer, dataset_kwargs)
    # the relations of PAS analysis come first and the one of coreference resolution last
    assert dataset.cohesion_tasks[0] == CohesionTask.PAS_ANALYSIS
    assert dataset.cohesion_tasks[-1] == CohesionTask.COREFERENCE_RESOLUTION
    kinds_of_pairs: set[str] = set()
    for i in range(len(dataset)):
        cohesion_mask = torch.as_tensor(dataset[i].cohesion_mask)
        assert torch.equal(cohesion_mask, _cohesion_mask_from_extractors(dataset, i))
        morphemes = dataset.doc_id2document[dataset.examples[i].doc_id].morphemes
        for anaphor in morphemes:
            for candidate in morphemes:
                if candidate.global_index < anaphor.global_index:
                    kinds_of_pairs.add("preceding")
                elif candidate.global_index > anaphor.global_index:
                    same_sentence = candidate.sentence.sid == anaphor.sentence.sid
                    kinds_of_pairs.add("following, same sentence" if same_sentence else "following, other sentence")
                    pair = (anaphor.global_index, candidate.global_index)
                    assert cohesion_mask[0][pair].item() is same_sentence
                    assert cohesion_mask[-1][pair].item() is False
    # the windows hold all three kinds of pairs the extractors tell apart
    assert kinds_of_pairs == {"preceding", "following, same sentence", "following, other sentence"}


class _SameSentencePasExtractor(PasExtractor):
    """A PasExtractor that only takes the other morphemes of the same sentence as candidates."""

    @staticmethod
    def is_candidate(unit: T, predicate: T) -> bool:
        return unit.sentence.sid == predicate.sentence.sid and unit.global_index != predicate.global_index


def test_cohesion_mask_of_an_extractor_with_its_own_rule(
    word_tokenizer: PreTrainedTokenizerBase, dataset_kwargs: dict[str, Any]
) -> None:
    dataset = _load_dataset(THREE_SENTENCES, word_tokenizer, dataset_kwargs)
    built_in_masks = [torch.as_tensor(dataset[i].cohesion_mask) for i in range(len(dataset))]
    dataset.cohesion_task2extractor[CohesionTask.PAS_ANALYSIS] = _SameSentencePasExtractor(
        list(dataset_kwargs["pas_cases"]),
        dataset.exophora_referent_types,
        verbal_predicate=True,
        nominal_predicate=True,
    )
    for i in range(len(dataset)):
        cohesion_mask = torch.as_tensor(dataset[i].cohesion_mask)
        assert torch.equal(cohesion_mask, _cohesion_mask_from_extractors(dataset, i))
    # the subclass's rule is not the one PasExtractor has, so the mask did change
    assert not torch.equal(cohesion_mask, built_in_masks[-1])
