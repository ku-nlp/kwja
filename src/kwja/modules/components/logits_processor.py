from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch
from transformers import PreTrainedTokenizerBase
from transformers.generation import LogitsProcessor

from kwja.utils.constants import FULL_SPACE_TOKEN, NEW_LINE_TOKEN


def get_char2tokens(tokenizer: PreTrainedTokenizerBase) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    char2tokens: Dict[str, Dict[str, int]] = defaultdict(dict)
    char2underscore_tokens: Dict[str, Dict[str, int]] = defaultdict(dict)
    for vocab_token, vocab_id in tokenizer.get_vocab().items():
        if vocab_token.startswith("▁"):
            if len(vocab_token) == 1:
                continue
            char2underscore_tokens[vocab_token[1]][vocab_token] = vocab_id
        else:
            char2tokens[vocab_token[0]][vocab_token] = vocab_id
    return char2tokens, char2underscore_tokens


class ForcedSurfLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        texts: List[str],
        num_beams: int,
        tokenizer: PreTrainedTokenizerBase,
        char2tokens: Dict[str, Dict[str, int]],
        char2underscore_tokens: Dict[str, Dict[str, int]],
    ) -> None:
        self.tokenizer = tokenizer
        self.texts: List[str] = []
        for text in tokenizer.batch_decode(tokenizer.batch_encode_plus(texts).input_ids):
            if text.endswith(self.tokenizer.eos_token):
                text = text[: -len(self.tokenizer.eos_token)]
            if f"{FULL_SPACE_TOKEN} " in text:
                text = text.replace(f"{FULL_SPACE_TOKEN} ", FULL_SPACE_TOKEN)
            self.texts.append(text)
        self.num_beams: int = num_beams
        self.char2tokens: Dict[str, Dict[str, int]] = char2tokens
        self.char2underscore_tokens: Dict[str, Dict[str, int]] = char2underscore_tokens
        self.new_line_token_id: int = tokenizer.convert_tokens_to_ids(NEW_LINE_TOKEN)
        self.underscore_token_id: int = tokenizer.convert_tokens_to_ids("▁")
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.is_decoding_surf: List[bool] = [True] * (len(self.texts) * self.num_beams)
        self.underscore_token_ids: Set[int] = {
            vocab_id for token in self.char2underscore_tokens.values() for vocab_id in token.values()
        } | {self.underscore_token_id}
        self.all_token_ids: Set[int] = {vocab_id for vocab_id in tokenizer.get_vocab().values()}

    def get_generated_surfs(self, input_ids: torch.Tensor) -> List[str]:
        generated_surfs: List[str] = []
        decodeds: List[str] = self.tokenizer.batch_decode(input_ids)
        for decoded in decodeds:
            generated_surf: str = ""
            for line in decoded.replace(NEW_LINE_TOKEN, "\n").split("\n"):
                stripped_line: str = line.lstrip().rstrip()
                if stripped_line in ["", "EOS"]:
                    break
                if stripped_line == "............/...":
                    generated_surf += "..."
                elif stripped_line[0] in ["!", "?", ",", "."]:
                    generated_surf += stripped_line[0]
                else:
                    split_line: List[str] = stripped_line.split()
                    if len(split_line) > 0:
                        generated_surf += split_line[0]
            generated_surfs.append(generated_surf)
        return generated_surfs

    def get_permitted_token_ids(self, text: str) -> Set[int]:
        permitted_token_ids: Set[int] = set()
        for vocab_token, vocab_id in self.char2tokens[text[0]].items():
            if text.startswith(vocab_token):
                permitted_token_ids.add(vocab_id)
        return permitted_token_ids

    def get_permitted_underscore_token_ids(self, text: str) -> Set[int]:
        permitted_underscore_token_ids: Set[int] = set()
        for vocab_token, vocab_id in self.char2underscore_tokens[text[0]].items():
            if text.startswith(vocab_token[1:]):
                permitted_underscore_token_ids.add(vocab_id)
        return permitted_underscore_token_ids

    def get_batch_banned_token_ids(self, prev_input_ids: torch.Tensor) -> List[List[int]]:
        banned_token_ids: List[List[int]] = []
        generated_surfs: List[str] = self.get_generated_surfs(prev_input_ids[:, 1:])
        for hypo_idx, input_ids in enumerate(prev_input_ids.tolist()):
            text: str = self.texts[hypo_idx // self.num_beams]
            generated_surf: str = generated_surfs[hypo_idx]
            if text == generated_surf:
                # 生成終了
                banned_token_ids.append([])
                continue

            if text.startswith(generated_surf):
                remaining_surf: str = text[len(generated_surf) :]
            else:
                # 生成されている文字列が，入力の先頭からの文字列とマッチしない場合は補正をしない
                banned_token_ids.append([])
                continue

            total_permitted_token_ids: Set[int] = set()
            if self.is_decoding_surf[hypo_idx] is True:
                if len(input_ids) == 1:
                    # 「<pad>」の次は，入力文字列の先頭からマッチするサブワードを許容
                    total_permitted_token_ids |= self.get_permitted_token_ids(text)
                    total_permitted_token_ids |= self.get_permitted_underscore_token_ids(text)
                    total_permitted_token_ids.add(self.underscore_token_id)
                elif len(input_ids) == 2:
                    if input_ids[-1] == self.underscore_token_id:
                        # 「<pad> "▁"」の次は，入力文字列の先頭からマッチするサブワードを許容．ただしアンダースコア始まりは許容しない
                        total_permitted_token_ids |= self.get_permitted_token_ids(remaining_surf)
                    else:
                        # 「<pad> "▁xxx"」または「<pad> "xxx"」の次は，入力文字列の先頭からマッチするサブワードを許容．また，全てのアンダースコア始まりのサブワードも許容
                        total_permitted_token_ids |= self.get_permitted_token_ids(remaining_surf)
                        total_permitted_token_ids |= self.underscore_token_ids
                elif (input_ids[-2], input_ids[-1]) == (self.new_line_token_id, self.underscore_token_id):
                    # 「改行 "▁"」の次は，まだ生成していない文字列の先頭からマッチするサブワードを許容．ただしアンダースコア始まりは許容しない
                    total_permitted_token_ids |= self.get_permitted_token_ids(remaining_surf)
                elif input_ids[-2] == self.new_line_token_id:
                    last_token: str = self.tokenizer.convert_ids_to_tokens(input_ids[-1])
                    if last_token.startswith("▁"):
                        # 「改行 "▁xxx"」の次は，まだ生成していない文字列の先頭からマッチするサブワードを許容．また，全てのアンダースコア始まりのサブワードも許容
                        total_permitted_token_ids |= self.get_permitted_token_ids(remaining_surf)
                        total_permitted_token_ids |= self.underscore_token_ids
                    else:
                        # 「改行 "xxx"」の次は，任意のサブワードを許容
                        pass
                elif input_ids[-1] == self.underscore_token_id:
                    self.is_decoding_surf[hypo_idx] = False
                    # 「"xxx" "▁"」 の次は，任意のサブワードを許容．ただしアンダースコア始まりは許容しない
                    total_permitted_token_ids |= self.all_token_ids - self.underscore_token_ids
                elif input_ids[-1] in self.underscore_token_ids:
                    self.is_decoding_surf[hypo_idx] = False
                    # 「"xxx" "▁yyy"」 の次は，任意のサブワードを許容
                    pass
                else:
                    # 「"xxx" "yyy"」 の次は，まだ生成していない文字列の先頭からマッチするサブワードを許容．また，全てのアンダースコア始まりのサブワードも許容
                    total_permitted_token_ids |= self.get_permitted_token_ids(remaining_surf)
                    total_permitted_token_ids |= self.underscore_token_ids
            else:
                if input_ids[-1] == self.new_line_token_id:
                    self.is_decoding_surf[hypo_idx] = True
                    # 「改行」の次は，まだ生成していない文字列の先頭からマッチするサブワードを許容
                    total_permitted_token_ids |= self.get_permitted_token_ids(remaining_surf)
                    total_permitted_token_ids |= self.get_permitted_underscore_token_ids(remaining_surf)
                    total_permitted_token_ids.add(self.underscore_token_id)
                else:
                    # surf のデコーディング時以外は，任意のサブワードを許容
                    pass

            if total_permitted_token_ids:
                banned_token_ids.append(
                    [i for i in self.tokenizer.get_vocab().values() if i not in total_permitted_token_ids]
                )
            else:
                banned_token_ids.append([])

        return banned_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        input_ids[input_ids == -100] = self.tokenizer.pad_token_id
        batch_banned_token_ids: List[List[int]] = self.get_batch_banned_token_ids(input_ids)
        for i, banned_token_ids in enumerate(batch_banned_token_ids):
            scores[i, banned_token_ids] = -float("inf")
        return scores
