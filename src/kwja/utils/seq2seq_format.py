from typing import Dict, List

from rhoknp import Sentence
from transformers import PreTrainedTokenizerBase

from kwja.utils.constants import (
    CANON_TOKEN,
    FULL_SPACE_TOKEN,
    HALF_SPACE_TOKEN1,
    HALF_SPACE_TOKEN2,
    LEMMA_TOKEN,
    NO_CANON_TOKEN,
    RARE_TO_SPECIAL,
    READING_TOKEN,
    SPECIAL_TO_RARE,
    SURF_TOKEN,
    TRIPLE_DOT_TOKEN,
)


class Seq2SeqFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.pad_token: str = self.tokenizer.pad_token

        self.word_to_token: Dict[str, str] = {
            "\u3000": FULL_SPACE_TOKEN,
            " ": HALF_SPACE_TOKEN1,
            "␣": HALF_SPACE_TOKEN2,
            "…": TRIPLE_DOT_TOKEN,
        }
        self.token_to_word: Dict[str, str] = {v: k for k, v in self.word_to_token.items()}

    def tokenize(self, mrph_lines: List[List[str]], tgt_mrphs: Dict[str, Dict[str, str]]) -> List[str]:
        is_partial: bool = len(tgt_mrphs) > 0
        output: List[str] = [self.pad_token] if is_partial else [SURF_TOKEN]
        for mrph_idx, mrph_line in enumerate(mrph_lines):
            tgt_mrph: Dict[str, str] = tgt_mrphs.get(str(mrph_idx), {})
            partial_anno_type: str = tgt_mrph.get("partial_annotation_type", "")  # {"", "canon", "norm"}

            if is_partial and partial_anno_type == "":
                special_tokens: List[str] = [self.pad_token, self.pad_token, self.pad_token, self.pad_token]
            else:
                special_tokens = [READING_TOKEN, LEMMA_TOKEN, CANON_TOKEN, SURF_TOKEN]
                if mrph_idx == len(mrph_lines) - 1:
                    special_tokens[-1] = self.tokenizer.eos_token

            for idx_in_mrph, mrph in enumerate(mrph_line):
                for k, v in RARE_TO_SPECIAL.items():
                    mrph = mrph.replace(k, v)
                tokenized: List[str] = [x for x in self.tokenizer.tokenize(mrph) if x != "▁"] + [
                    special_tokens[idx_in_mrph]
                ]
                if is_partial:
                    if partial_anno_type == "canon" or (partial_anno_type == "norm" and idx_in_mrph in {0, 2}):
                        output.extend(tokenized)
                    else:
                        output.extend([self.pad_token] * len(tokenized))
                else:
                    output.extend(tokenized)
        return output

    def sent_to_text(self, sentence: Sentence) -> str:
        text: str = sentence.text
        for k, v in self.word_to_token.items():
            text = text.replace(k, v)
        text = text.replace(HALF_SPACE_TOKEN2, HALF_SPACE_TOKEN1)
        for k, v in RARE_TO_SPECIAL.items():
            text = text.replace(k, v)

        tokenized: List[str] = [token for token in self.tokenizer.tokenize(text) if token != "▁"]
        decoded: str = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokenized))
        for token in [FULL_SPACE_TOKEN, HALF_SPACE_TOKEN1, HALF_SPACE_TOKEN2, TRIPLE_DOT_TOKEN]:
            decoded = decoded.replace(f"{token} ", token)
        for token in SPECIAL_TO_RARE:
            decoded = decoded.replace(f"{token} ", token)
        decoded = decoded.replace(" ", HALF_SPACE_TOKEN1)
        return decoded

    @staticmethod
    def sent_to_mrph_lines(sentence: Sentence) -> List[List[str]]:
        outputs: List[List[str]] = []
        for morpheme in sentence.morphemes:
            if morpheme.surf == "\u3000":
                surf: str = FULL_SPACE_TOKEN
                reading: str = FULL_SPACE_TOKEN
                lemma: str = FULL_SPACE_TOKEN
                canon: str = "/"
            elif morpheme.surf == " ":
                surf = HALF_SPACE_TOKEN1
                reading = HALF_SPACE_TOKEN1
                lemma = HALF_SPACE_TOKEN1
                canon = "/"
            elif morpheme.surf == "…":
                surf = TRIPLE_DOT_TOKEN
                reading = TRIPLE_DOT_TOKEN
                lemma = TRIPLE_DOT_TOKEN
                canon = f"{TRIPLE_DOT_TOKEN}/{TRIPLE_DOT_TOKEN}"
            else:
                surf = morpheme.surf
                if morpheme.reading == "\u3000":
                    reading = FULL_SPACE_TOKEN
                elif "/" in morpheme.reading and len(morpheme.reading) > 1:
                    reading = morpheme.reading.split("/")[0]
                else:
                    reading = morpheme.reading
                lemma = FULL_SPACE_TOKEN if morpheme.lemma == "\u3000" else morpheme.lemma
                if morpheme.canon is not None:
                    canon = morpheme.canon
                    canon_list: List[str] = canon.split("/")
                    if len(canon_list) > 2 and canon_list[0] and canon_list[1]:
                        canon = f"{canon_list[0]}/{canon_list[1]}"
                else:
                    canon = NO_CANON_TOKEN
            outputs.append([surf, reading, lemma, canon])
        return outputs

    def format_to_sent(self, text: str) -> Sentence:
        lines: List[str] = text.split(SURF_TOKEN)
        formatted: str = ""
        for line in lines:
            if not line:
                continue
            try:
                surf: str = line.split(READING_TOKEN)[0]
                surf = self.token_to_word[surf] if surf in self.token_to_word else surf
                for k, v in SPECIAL_TO_RARE.items():
                    surf = surf.replace(k, v)

                reading: str = line.split(READING_TOKEN)[1].split(LEMMA_TOKEN)[0]
                reading = self.token_to_word[reading] if reading in self.token_to_word else reading
                for k, v in SPECIAL_TO_RARE.items():
                    reading = reading.replace(k, v)

                lemma: str = line.split(LEMMA_TOKEN)[1].split(CANON_TOKEN)[0]
                lemma = self.token_to_word[lemma] if lemma in self.token_to_word else lemma
                for k, v in SPECIAL_TO_RARE.items():
                    lemma = lemma.replace(k, v)

                canon: str = line.split(CANON_TOKEN)[1]
                for k, v in self.token_to_word.items():
                    canon = canon.replace(k, v)
                for k, v in SPECIAL_TO_RARE.items():
                    canon = canon.replace(k, v)
                canon = (
                    f"{self.token_to_word[TRIPLE_DOT_TOKEN]}/{self.token_to_word[TRIPLE_DOT_TOKEN]}"
                    if canon == f"{TRIPLE_DOT_TOKEN}/{TRIPLE_DOT_TOKEN}"
                    else canon
                )
                canon = f'"代表表記:{canon}"' if canon != NO_CANON_TOKEN else "NIL"

                # 例外処理
                if surf == " " and reading == "\u3000" and lemma == "\u3000":
                    surf = "\u3000"
                if surf == "°C":
                    surf, lemma, canon = "℃", "℃", '"代表表記:℃/ど"'

                formatted += f"{surf} {reading} {lemma} 未定義語 15 その他 1 * 0 * 0 {canon}\n"
            except IndexError:
                formatted += "@ @ @ 未定義語 15 その他 1 * 0 * 0 NIL\n"
        formatted += "EOS\n"
        return Sentence.from_jumanpp(formatted)
