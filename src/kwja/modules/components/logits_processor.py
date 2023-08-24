from copy import deepcopy
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
        texts: List[str],
        num_beams: int,
        tokenizer: PreTrainedTokenizerFast,
        reading_candidates: Set[int],
        char2tokens: Dict[str, Dict[str, int]],
    ) -> None:
        self.tokenizer = tokenizer
        self.texts: List[str] = texts
        self.num_beams: int = num_beams
        self.char2tokens: Dict[str, Dict[str, int]] = char2tokens
        self.pad_token_id: int = self.tokenizer.pad_token_id

        self.surf_token_id: int = tokenizer.convert_tokens_to_ids(SURF_TOKEN)
        self.reading_token_id: int = tokenizer.convert_tokens_to_ids(READING_TOKEN)
        self.lemma_token_id: int = tokenizer.convert_tokens_to_ids(LEMMA_TOKEN)
        self.canon_token_id: int = tokenizer.convert_tokens_to_ids(CANON_TOKEN)

        self.all_token_ids: Set[int] = set(self.tokenizer.get_vocab().values())
        self.ids_except_surf: List[int] = list(self.all_token_ids - {self.surf_token_id})
        self.ids_except_kanji_and_katakana: Set[int] = deepcopy(self.all_token_ids) - reading_candidates

        self.token_to_ids_except_token: Dict[str, Set[int]] = {}
        special_tokens: List[str] = [FULL_SPACE_TOKEN, HALF_SPACE_TOKEN1, HALF_SPACE_TOKEN2, TRIPLE_DOT_TOKEN] + list(
            SPECIAL_TO_RARE.keys()
        )
        for special_token in special_tokens:
            self.token_to_ids_except_token[special_token] = deepcopy(self.all_token_ids) - {
                self.tokenizer.convert_tokens_to_ids(special_token)
            }
            self.token_to_ids_except_token[special_token] = deepcopy(self.all_token_ids) - {
                self.tokenizer.convert_tokens_to_ids(special_token)
            }

    def get_generated_surfs(self, input_ids: torch.Tensor) -> List[str]:
        generated_surfs: List[str] = []
        decodeds: List[str] = self.tokenizer.batch_decode(input_ids)
        for decoded in decodeds:
            generated_surf: str = ""
            for line in decoded.split(SURF_TOKEN):
                generated_surf += line.split(READING_TOKEN)[0].strip(" ")
            generated_surfs.append(generated_surf)
        return generated_surfs

    def get_banned_token_ids(self, text: str) -> Set[int]:
        for token, ids_except_token in self.token_to_ids_except_token.items():
            if text.startswith(token):
                return ids_except_token
        permitted_token_ids: Set[int] = set()
        for vocab_token, vocab_id in self.char2tokens[text[0]].items():
            if (vocab_token.startswith("▁") and text.startswith(vocab_token[1:])) or text.startswith(vocab_token):
                permitted_token_ids.add(vocab_id)
        return deepcopy(self.all_token_ids) - permitted_token_ids

    def get_target_morpheme(self, input_ids: List[int]) -> TargetMorpheme:
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

    def get_mask(self, prev_input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor = torch.zeros_like(scores).bool()
        _, seq_len = prev_input_ids.size()
        if seq_len == 1:
            mask[:, self.ids_except_surf] = True
            return mask

        generated_surfs: List[str] = self.get_generated_surfs(prev_input_ids[:, 2:])
        for hypo_idx, input_ids in enumerate(prev_input_ids.tolist()):
            target_morpheme: TargetMorpheme = self.get_target_morpheme(input_ids)

            text: str = self.texts[hypo_idx // self.num_beams]
            generated_surf: str = generated_surfs[hypo_idx]
            if text.startswith(generated_surf):
                remaining_surf: str = text[len(generated_surf) :]
            else:
                # 生成されている文字列が，入力の先頭からの文字列とマッチしない場合は補正をしない
                continue

            banned_token_ids: Set[int] = set()
            if target_morpheme.surf is True:
                if remaining_surf:
                    banned_token_ids |= self.get_banned_token_ids(remaining_surf)
                    if input_ids[-1] != self.surf_token_id:
                        banned_token_ids.discard(self.reading_token_id)
                elif input_ids[-1] == self.surf_token_id:
                    banned_token_ids.add(self.reading_token_id)
            elif target_morpheme.reading is True:
                banned_token_ids |= self.ids_except_kanji_and_katakana
                if input_ids[-1] == self.reading_token_id:
                    banned_token_ids.add(self.lemma_token_id)
                else:
                    banned_token_ids.discard(self.lemma_token_id)
            elif target_morpheme.lemma is True:
                if input_ids[-1] == self.lemma_token_id:
                    banned_token_ids.add(self.canon_token_id)
            elif target_morpheme.canon is True:
                if (not remaining_surf) or (input_ids[-1] == self.canon_token_id):
                    banned_token_ids.add(self.surf_token_id)

            if remaining_surf or target_morpheme.canon is False:
                banned_token_ids.add(self.tokenizer.eos_token_id)
            else:
                banned_token_ids.discard(self.tokenizer.eos_token_id)
            mask[hypo_idx, list(banned_token_ids)] = True
        return mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        input_ids[input_ids == -100] = self.tokenizer.pad_token_id
        mask: torch.Tensor = self.get_mask(input_ids, scores)
        scores.masked_fill_(mask, -float("inf"))
        return scores
