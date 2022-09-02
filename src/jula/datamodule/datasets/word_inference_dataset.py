from collections import defaultdict

import torch
from omegaconf import ListConfig
from rhoknp import Document, Morpheme, Sentence
from rhoknp.cohesion import ExophoraReferent
from rhoknp.props import FeatureDict, SemanticsDict
from rhoknp.units.morpheme import MorphemeAttributes
from tokenizers import Encoding
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from jula.datamodule.examples import CohesionTask
from jula.datamodule.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor


class WordInferenceDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        pas_cases: ListConfig,
        bar_rels: ListConfig,
        exophora_referents: ListConfig,
        cohesion_tasks: ListConfig,
        special_tokens: ListConfig,
        restrict_cohesion_target: bool,
        model_name_or_path: str = "nlp-waseda/roberta-base-japanese",
        max_seq_length: int = 512,
        tokenizer_kwargs: dict = None,
        **_,
    ) -> None:
        did2sentences = defaultdict(list)
        doc_id, sid = "", ""
        for text in texts:
            if text.startswith("#"):
                sentence = Sentence.from_raw_text(text)
                doc_id, sid = sentence.doc_id, sentence.sid
            else:
                sentence = Sentence()
                sentence.doc_id, sentence.sid = doc_id, sid
                morphemes = []
                for word in text.split(" "):
                    attributes = MorphemeAttributes(
                        surf=word,
                        reading="null",
                        lemma="null",
                        pos="null",
                        pos_id=0,
                        subpos="null",
                        subpos_id=0,
                        conjtype="null",
                        conjtype_id=0,
                        conjform="null",
                        conjform_id=0,
                    )
                    morphemes.append(Morpheme(attributes, SemanticsDict(), FeatureDict()))
                sentence.morphemes = morphemes
                did2sentences[doc_id].append(sentence)
        self.documents = [Document.from_sentences(sentences) for sentences in did2sentences.values()]

        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.special_tokens: list[str] = list(special_tokens)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **(tokenizer_kwargs or {}),
        )
        self.max_seq_length = max_seq_length
        self.special_to_index: dict[str, int] = {
            token: self.max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        self.index_to_special: dict[int, str] = {v: k for k, v in self.special_to_index.items()}
        self.cohesion_tasks = [CohesionTask(t) for t in cohesion_tasks]
        self.pas_cases: list[str] = list(pas_cases)
        self.bar_rels: list[str] = list(bar_rels)
        self.cohesion_task_to_rel_types = {
            CohesionTask.PAS_ANALYSIS: self.pas_cases,
            CohesionTask.BRIDGING: self.bar_rels,
            CohesionTask.COREFERENCE: ["="],
        }
        self.extractors = {
            CohesionTask.PAS_ANALYSIS: PasExtractor(self.pas_cases, self.exophora_referents, restrict_cohesion_target),
            CohesionTask.COREFERENCE: CoreferenceExtractor(self.exophora_referents, restrict_cohesion_target),
            CohesionTask.BRIDGING: BridgingExtractor(self.bar_rels, self.exophora_referents, restrict_cohesion_target),
        }
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

    @property
    def special_indices(self) -> list[int]:
        return list(self.special_to_index.values())

    @property
    def cohesion_special_indices(self) -> list[int]:
        return [index for special, index in self.special_to_index.items() if special != "[ROOT]"]

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def cohesion_rel_types(self) -> list[str]:
        return [t for ts in self.cohesion_task_to_rel_types.values() for t in ts]

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.documents[index], index)

    def encode(self, document: Document, example_id: int) -> dict[str, torch.Tensor]:
        encoding: Encoding = self.tokenizer(
            " ".join(m.text for m in document.morphemes),
            truncation=True,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=self.max_seq_length - self.num_special_tokens,
        ).encodings[0]

        dependency_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for sentence in document.sentences:
            num_intra_morphemes = len(sentence.morphemes)
            for i in range(num_intra_morphemes):
                for j in range(num_intra_morphemes):
                    if i != j:
                        dependency_mask[i][j] = True
                dependency_mask[i][self.special_to_index["[ROOT]"]] = True

        cohesion_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        num_morphemes = len(document.morphemes)
        for i in range(num_morphemes):
            for j in range(num_morphemes):
                if i != j:
                    cohesion_mask[i][j] = True
            for special_index in self.cohesion_special_indices:
                cohesion_mask[i][special_index] = True

        merged_encoding: Encoding = Encoding.merge([encoding, self.special_encoding])

        return {
            "example_ids": torch.tensor(example_id, dtype=torch.long),
            "input_ids": torch.tensor(merged_encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(merged_encoding.attention_mask, dtype=torch.long),
            "subword_map": torch.tensor(self._gen_subword_map(merged_encoding), dtype=torch.bool),
            "reading_subword_map": torch.tensor(
                self._gen_subword_map(merged_encoding, include_additional_words=False), dtype=torch.bool
            ),
            "intra_mask": torch.tensor(dependency_mask, dtype=torch.bool),
            "cohesion_mask": torch.tensor(cohesion_mask, dtype=torch.bool)
            .unsqueeze(0)
            .expand(len(self.cohesion_rel_types), self.max_seq_length, self.max_seq_length),
            "tokens": " ".join(self.tokenizer.decode(id_) for id_ in merged_encoding.ids),
        }

    def _gen_subword_map(self, encoding: Encoding, include_additional_words: bool = True) -> list[list[bool]]:
        subword_map = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for token_id, word_id in enumerate(encoding.word_ids):
            if word_id is not None:
                subword_map[word_id][token_id] = True
        if include_additional_words:
            for special_index in self.special_indices:
                subword_map[special_index][special_index] = True
        return subword_map
