from dataclasses import dataclass

import regex
import torch
from transformers import PreTrainedTokenizerFast
from transformers.generation import LogitsProcessor

from kwja.modules.functions.loss import mask_logits
from kwja.utils.constants import (
    CANON_TOKEN,
    HALF_SPACE_TOKEN,
    LEMMA_TOKEN,
    MORPHEME_DELIMITER_TOKEN,
    NO_CANON_TOKEN,
    READING_TOKEN,
    SPECIAL2RARE,
    SURF_TOKEN,
)

KANJI_KATAKANA_PAT = r"[\p{Script=Han}\p{Script=Katakana}]"


def get_reading_candidate_token_ids(tokenizer: PreTrainedTokenizerFast) -> list[int]:
    control_tokens = {
        tokenizer.pad_token,
        tokenizer.eos_token,
        SURF_TOKEN,
        READING_TOKEN,
        LEMMA_TOKEN,
        CANON_TOKEN,
        NO_CANON_TOKEN,
        MORPHEME_DELIMITER_TOKEN,
    }
    return [
        token_id
        for token, token_id in tokenizer.vocab.items()
        # 漢字またはカタカナを含むトークンIDは除外（読みには漢字とカタカナが含まれないので）
        if regex.search(KANJI_KATAKANA_PAT, token) is None and token not in control_tokens
    ]


def get_char2token_items(tokenizer: PreTrainedTokenizerFast) -> dict[str, dict[str, int]]:
    char2token_items: dict[str, dict[str, int]] = {}
    for token, token_id in tokenizer.vocab.items():
        if token.startswith("▁"):
            if len(token) == 1:
                continue
            char: str = token[1]
        else:
            char = token[0]
        char2token_items.setdefault(char, {})
        char2token_items[char][token] = token_id
    return char2token_items


@dataclass
class TargetProperty:
    surf: bool = False
    reading: bool = False
    lemma: bool = False
    canon: bool = False


class SurfForcedDecodingLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        batch_surfs: list[list[str]],
        num_beams: int,
        tokenizer: PreTrainedTokenizerFast,
        char2token_items: dict[str, dict[str, int]],
        reading_candidate_token_ids: list[int],
    ) -> None:
        self.batch_surfs: list[list[str]] = batch_surfs
        self.num_beams: int = num_beams

        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab  # cache

        self.char2token_items: dict[str, dict[str, int]] = char2token_items
        self.reading_candidate_token_ids: list[int] = reading_candidate_token_ids

        self.is_finished: list[bool] = [False] * len(batch_surfs)

    def __call__(self, batch_prev_input_ids: torch.Tensor, batch_logits: torch.Tensor) -> torch.Tensor:
        # Falseならば-inf（-infでないと対応するトークンが選ばれてしまう可能性がある）
        mask = self.get_mask(batch_prev_input_ids, batch_logits)
        batch_masked_logits = mask_logits(batch_logits, mask, mask_value=-float("inf"))
        return batch_masked_logits

    def get_mask(
        self,
        batch_prev_input_ids: torch.Tensor,  # (b * num_beams, seq_len)
        batch_logits: torch.Tensor,  # (b * num_beams, vocab_size)
    ) -> torch.Tensor:
        _, seq_len = batch_prev_input_ids.size()
        if seq_len == 1:
            mask = torch.zeros_like(batch_logits, dtype=torch.bool)
            mask[:, self.vocab[SURF_TOKEN]] = True
            return mask
        else:
            batch_masks = []
            for i, (prev_input_ids, logits) in enumerate(zip(batch_prev_input_ids.tolist(), batch_logits)):
                batch_idx = i // self.num_beams
                if self.is_finished[batch_idx]:
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask[self.tokenizer.eos_token_id] = True
                    batch_masks.append(mask)
                    continue
                target_property: TargetProperty = self._get_target_property(prev_input_ids)
                if target_property.surf is True:
                    batch_masks.append(self._get_surf_mask(prev_input_ids, logits, batch_idx))
                elif target_property.reading is True:
                    batch_masks.append(self._get_reading_mask(prev_input_ids, logits))
                elif target_property.lemma is True:
                    batch_masks.append(self._get_lemma_mask(prev_input_ids, logits))
                elif target_property.canon is True:
                    batch_masks.append(self._get_canon_mask(prev_input_ids, logits, batch_idx))
            return torch.stack(batch_masks)

    def _get_target_property(self, prev_input_ids: list[int]) -> TargetProperty:
        target_property = TargetProperty()
        for prev_input_id in prev_input_ids[::-1]:
            if prev_input_id == self.vocab[SURF_TOKEN]:
                target_property.surf = True
                break
            elif prev_input_id == self.vocab[READING_TOKEN]:
                target_property.reading = True
                break
            elif prev_input_id == self.vocab[LEMMA_TOKEN]:
                target_property.lemma = True
                break
            elif prev_input_id == self.vocab[CANON_TOKEN]:
                target_property.canon = True
                break
        return target_property

    def _get_surf_mask(self, prev_input_ids: list[int], logits: torch.Tensor, batch_idx: int) -> torch.Tensor:
        surf_mask = torch.zeros_like(logits, dtype=torch.bool)
        if ungenerated_surf := self._get_ungenerated_surf(prev_input_ids, self.batch_surfs[batch_idx]):
            surf_mask[self._get_permitted_token_ids(ungenerated_surf)] = True
        else:
            surf_mask[self.vocab[READING_TOKEN]] = True
        return surf_mask

    def _get_reading_mask(self, prev_input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        reading_mask = torch.zeros_like(logits, dtype=torch.bool)
        if prev_input_ids[-1] == self.vocab[READING_TOKEN]:
            reading_mask[self.reading_candidate_token_ids] = True
        else:
            reading_mask[[*self.reading_candidate_token_ids, self.vocab[LEMMA_TOKEN]]] = True
        return reading_mask

    def _get_lemma_mask(self, prev_input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        lemma_mask = torch.ones_like(logits, dtype=torch.bool)
        prohibited_token_ids = [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.vocab[SURF_TOKEN],
            self.vocab[READING_TOKEN],
            self.vocab[LEMMA_TOKEN],
            self.vocab[NO_CANON_TOKEN],
            self.vocab[MORPHEME_DELIMITER_TOKEN],
        ]
        if prev_input_ids[-1] == self.vocab[LEMMA_TOKEN]:
            prohibited_token_ids.append(self.vocab[CANON_TOKEN])
        lemma_mask[prohibited_token_ids] = False
        return lemma_mask

    def _get_canon_mask(self, prev_input_ids: list[int], logits: torch.Tensor, batch_idx: int) -> torch.Tensor:
        canon_mask = torch.ones_like(logits, dtype=torch.bool)
        prohibited_token_ids = [
            self.tokenizer.pad_token_id,
            self.vocab[READING_TOKEN],
            self.vocab[LEMMA_TOKEN],
            self.vocab[CANON_TOKEN],
            self.vocab[MORPHEME_DELIMITER_TOKEN],
        ]
        if prev_input_ids[-1] == self.vocab[CANON_TOKEN]:
            prohibited_token_ids += [self.tokenizer.eos_token_id, self.vocab[SURF_TOKEN]]
        elif prev_input_ids.count(self.vocab[READING_TOKEN]) < len(self.batch_surfs[batch_idx]):
            prohibited_token_ids += [self.tokenizer.eos_token_id, self.vocab[NO_CANON_TOKEN]]
        else:
            prohibited_token_ids += [self.vocab[SURF_TOKEN], self.vocab[NO_CANON_TOKEN]]
            if logits.argmax().item() == self.tokenizer.eos_token_id:
                self.is_finished[batch_idx] = True
        canon_mask[prohibited_token_ids] = False
        return canon_mask

    def _get_ungenerated_surf(self, prev_input_ids: list[int], surfs: list[str]) -> str:
        decoded: str = self.tokenizer.decode(prev_input_ids)
        surf_index: int = 0
        generated_surf: str = ""
        for line in decoded.split(SURF_TOKEN):
            if READING_TOKEN in line:
                surf_index += 1
            else:
                generated_surf = line.strip(" ")
        return surfs[surf_index][len(generated_surf) :]

    def _get_permitted_token_ids(self, ungenerated_surf: str) -> list[int]:
        for special_token in [HALF_SPACE_TOKEN, *list(SPECIAL2RARE.keys())]:
            if ungenerated_surf.startswith(special_token):
                return [self.vocab[special_token]]

        permitted_token_ids: list[int] = []
        for token, token_id in self.char2token_items[ungenerated_surf[0]].items():
            if ungenerated_surf.startswith(token) or (token.startswith("▁") and ungenerated_surf.startswith(token[1:])):
                permitted_token_ids.append(token_id)
        return permitted_token_ids
