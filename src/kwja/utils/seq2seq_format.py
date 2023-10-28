from typing import Dict, List

from rhoknp import Sentence
from transformers import PreTrainedTokenizerFast

from kwja.utils.constants import (
    CANON_TOKEN,
    FULL_SPACE_TOKEN,
    HALF_SPACE_TOKEN1,
    HALF_SPACE_TOKEN2,
    LEMMA_TOKEN,
    MORPHEME_SPLIT_TOKEN,
    NO_CANON_TOKEN,
    RARE_TO_SPECIAL,
    READING_TOKEN,
    SPECIAL_TO_RARE,
    SURF_TOKEN,
    TRIPLE_DOT_TOKEN,
)


class Seq2SeqFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer: PreTrainedTokenizerFast = tokenizer

        self.word_to_token: Dict[str, str] = {
            "\u3000": FULL_SPACE_TOKEN,
            " ": HALF_SPACE_TOKEN1,
            "␣": HALF_SPACE_TOKEN2,
            "…": TRIPLE_DOT_TOKEN,
        }
        self.token_to_word: Dict[str, str] = {v: k for k, v in self.word_to_token.items()}

    def get_surfs(self, sentence: Sentence) -> List[str]:
        surfs: List[str] = []
        for morpheme in sentence.morphemes:
            surf: str = morpheme.surf
            for k, v in self.word_to_token.items():
                surf = surf.replace(k, v)
            surf = surf.replace(HALF_SPACE_TOKEN2, HALF_SPACE_TOKEN1)
            for k, v in RARE_TO_SPECIAL.items():
                surf = surf.replace(k, v)
            tokenized_surf: List[str] = [x for x in self.tokenizer.tokenize(surf) if x != "▁"]
            decoded: str = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokenized_surf))
            for token in self.word_to_token.values():
                decoded = decoded.replace(f"{token} ", token)
            for token in SPECIAL_TO_RARE:
                decoded = decoded.replace(f"{token} ", token)
            surfs.append(decoded.replace(" ", HALF_SPACE_TOKEN1))
        return surfs

    def get_src_tokens(self, sentence: Sentence) -> List[str]:
        src_text: str = MORPHEME_SPLIT_TOKEN.join(m.surf for m in sentence.morphemes)
        for k, v in self.word_to_token.items():
            src_text = src_text.replace(k, v)
        src_text = src_text.replace(HALF_SPACE_TOKEN2, HALF_SPACE_TOKEN1)
        for k, v in RARE_TO_SPECIAL.items():
            src_text = src_text.replace(k, v)
        return [x for x in self.tokenizer.tokenize(src_text) if x != "▁"]

    def get_tgt_tokens(self, sentence: Sentence) -> List[str]:
        seq2seq_format: str = ""
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
            seq2seq_format += f"{SURF_TOKEN}{surf}{READING_TOKEN}{reading}{LEMMA_TOKEN}{lemma}{CANON_TOKEN}{canon}"
        for k, v in RARE_TO_SPECIAL.items():
            seq2seq_format = seq2seq_format.replace(k, v)
        return [x for x in self.tokenizer.tokenize(seq2seq_format) if x != "▁"]

    def format_to_sent(self, text: str) -> Sentence:
        formatted: str = ""
        for line in text.split(SURF_TOKEN):
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
