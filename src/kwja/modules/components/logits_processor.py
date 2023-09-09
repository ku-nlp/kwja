from dataclasses import dataclass
from typing import Dict, List, Set

import regex
import torch
from transformers import PreTrainedTokenizerFast
from transformers.generation import LogitsProcessor

from kwja.utils.constants import (
    CANON_TOKEN,
    FULL_SPACE_TOKEN,
    HALF_SPACE_TOKEN1,
    HALF_SPACE_TOKEN2,
    LEMMA_TOKEN,
    READING_TOKEN,
    SPECIAL_TO_RARE,
    SURF_TOKEN,
    TRIPLE_DOT_TOKEN,
)

KANJI_KATAKANA_PATTERN = r"[\p{Script=Han}\p{Script=Katakana}]"


@dataclass()
class TargetMorpheme:
    surf: bool = False
    reading: bool = False
    lemma: bool = False
    canon: bool = False


def get_reading_candidates(tokenizer: PreTrainedTokenizerFast) -> Set[int]:
    candidates: Set[int] = set()
    for token, vocab_id in tokenizer.vocab.items():
        if not bool(regex.search(KANJI_KATAKANA_PATTERN, token)):
            # 漢字またはカタカナを含む語彙は読みからは除外（＝読みには漢字とカタカナが含まれない）
            candidates.add(vocab_id)
    return candidates


def get_char2tokens(tokenizer: PreTrainedTokenizerFast) -> Dict[str, Dict[str, int]]:
    char2tokens: Dict[str, Dict[str, int]] = {}
    for vocab_token, vocab_id in tokenizer.get_vocab().items():
        if vocab_token.startswith("▁"):
            if len(vocab_token) == 1:
                continue
            char: str = vocab_token[1]
        else:
            char = vocab_token[0]
        if char not in char2tokens:
            char2tokens[char] = {}
        char2tokens[char][vocab_token] = vocab_id
    return char2tokens


class ForcedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        surfs: List[List[str]],
        num_beams: int,
        tokenizer: PreTrainedTokenizerFast,
        reading_candidates: Set[int],
        char2tokens: Dict[str, Dict[str, int]],
    ) -> None:
        self.tokenizer = tokenizer
        self.surfs: List[List[str]] = surfs
        self.num_beams: int = num_beams
        self.char2tokens: Dict[str, Dict[str, int]] = char2tokens
        self.eos_token_id: int = self.tokenizer.eos_token_id

        self.surf_token_id: int = tokenizer.convert_tokens_to_ids(SURF_TOKEN)
        self.reading_token_id: int = tokenizer.convert_tokens_to_ids(READING_TOKEN)
        self.lemma_token_id: int = tokenizer.convert_tokens_to_ids(LEMMA_TOKEN)
        self.canon_token_id: int = tokenizer.convert_tokens_to_ids(CANON_TOKEN)

        self.ids_except_surf: List[int] = list(set(self.tokenizer.get_vocab().values()) - {self.surf_token_id})
        self.ids_except_reading: Set[int] = set(self.tokenizer.get_vocab().values()) - {self.reading_token_id}
        self.ids_except_kanji_and_katakana: Set[int] = set(self.tokenizer.get_vocab().values()) - reading_candidates

        self.token_to_ids_except_token: Dict[str, Set[int]] = {}
        special_tokens: List[str] = [FULL_SPACE_TOKEN, HALF_SPACE_TOKEN1, HALF_SPACE_TOKEN2, TRIPLE_DOT_TOKEN] + list(
            SPECIAL_TO_RARE.keys()
        )
        for special_token in special_tokens:
            self.token_to_ids_except_token[special_token] = set(self.tokenizer.get_vocab().values()) - {
                self.tokenizer.convert_tokens_to_ids(special_token)
            }
        self.is_finished: List[bool] = [False] * len(self.surfs)

    def _get_target_morpheme(self, input_ids: List[int]) -> TargetMorpheme:
        target_morpheme: TargetMorpheme = TargetMorpheme()
        for input_id in input_ids[::-1]:
            if input_id == self.surf_token_id:
                target_morpheme.surf = True
                break
            elif input_id == self.reading_token_id:
                target_morpheme.reading = True
                break
            elif input_id == self.lemma_token_id:
                target_morpheme.lemma = True
                break
            elif input_id == self.canon_token_id:
                target_morpheme.canon = True
                break
        return target_morpheme

    def _get_remaining_surf(self, input_ids: List[int], surf: List[str]) -> str:
        decoded: str = self.tokenizer.decode(input_ids)
        surf_index: int = 0
        generated_surf: str = ""
        for line in decoded.split(SURF_TOKEN):
            if READING_TOKEN in line:
                surf_index += 1
            else:
                generated_surf = line.strip(" ")
        return surf[surf_index][len(generated_surf) :]

    def _get_banned_token_ids(self, text: str) -> Set[int]:
        for token, ids_except_token in self.token_to_ids_except_token.items():
            if text.startswith(token):
                return ids_except_token
        permitted_token_ids: Set[int] = set()
        for vocab_token, vocab_id in self.char2tokens[text[0]].items():
            if (vocab_token.startswith("▁") and text.startswith(vocab_token[1:])) or text.startswith(vocab_token):
                permitted_token_ids.add(vocab_id)
        return set(self.tokenizer.get_vocab().values()) - permitted_token_ids

    def _get_generated_surf(self, input_ids: List[int]) -> List[str]:
        decoded: str = self.tokenizer.decode(input_ids)
        generated_surf: List[str] = []
        for line in decoded.split(SURF_TOKEN)[1:]:
            generated_surf.append(line.split(READING_TOKEN)[0].strip(" "))
        return generated_surf

    def get_mask(self, prev_input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor = torch.zeros_like(scores).bool()
        _, seq_len = prev_input_ids.size()
        if seq_len == 1:
            mask[:, self.ids_except_surf] = True
            return mask

        for hypo_idx, input_ids in enumerate(prev_input_ids.tolist()):
            if self.is_finished[hypo_idx // self.num_beams]:
                continue
            target_morpheme: TargetMorpheme = self._get_target_morpheme(input_ids)

            banned_token_ids: Set[int] = {
                self.eos_token_id,
                self.surf_token_id,
                self.reading_token_id,
                self.lemma_token_id,
                self.canon_token_id,
            }
            if target_morpheme.surf:
                if remaining_surf := self._get_remaining_surf(input_ids, self.surfs[hypo_idx // self.num_beams]):
                    banned_token_ids |= self._get_banned_token_ids(remaining_surf)
                else:
                    banned_token_ids = self.ids_except_reading
            elif target_morpheme.reading:
                banned_token_ids |= self.ids_except_kanji_and_katakana
                if input_ids[-1] != self.reading_token_id:
                    banned_token_ids.discard(self.lemma_token_id)
            elif target_morpheme.lemma:
                if input_ids[-1] != self.lemma_token_id:
                    banned_token_ids.discard(self.canon_token_id)
            elif target_morpheme.canon:
                if input_ids[-1] != self.canon_token_id:
                    generated_surf: List[str] = self._get_generated_surf(input_ids)
                    if len(generated_surf) == len(self.surfs[hypo_idx // self.num_beams]):
                        banned_token_ids.discard(self.eos_token_id)
                        self.is_finished[hypo_idx // self.num_beams] = True
                    else:
                        banned_token_ids.discard(self.surf_token_id)
            mask[hypo_idx, list(banned_token_ids)] = True
        return mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        input_ids[input_ids == -100] = self.tokenizer.pad_token_id
        mask: torch.Tensor = self.get_mask(input_ids, scores)
        scores.masked_fill_(mask, -float("inf"))
        return scores
