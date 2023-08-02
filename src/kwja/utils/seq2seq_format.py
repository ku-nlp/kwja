from typing import Dict, List

from rhoknp import Sentence
from transformers import PreTrainedTokenizerBase

from kwja.utils.constants import (
    CANON_TOKEN,
    FULL_SPACE_TOKEN,
    HALF_SPACE_TOKEN,
    LEMMA_TOKEN,
    NO_CANON_TOKEN,
    READING_TOKEN,
    SURF_TOKEN,
    TRIPLE_DOT_TOKEN,
)


class Seq2SeqFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

        self.word_to_token: Dict[str, str] = {
            "\u3000": FULL_SPACE_TOKEN,
            " ": HALF_SPACE_TOKEN,
            "…": TRIPLE_DOT_TOKEN,
        }
        self.token_to_word: Dict[str, str] = {v: k for k, v in self.word_to_token.items()}

    def tokenize(self, texts: List[str]) -> List[str]:
        tokenized: List[str] = []
        for text in texts:
            tokenized_tmp: List[str] = self.tokenizer.tokenize(text)
            if len(tokenized_tmp) > 1 and tokenized_tmp[0] == "▁":
                tokenized_tmp = tokenized_tmp[1:]
            tokenized.extend(tokenized_tmp)
        return tokenized

    def sent_to_text(self, sentence: Sentence) -> str:
        text: str = sentence.text
        for k, v in self.word_to_token.items():
            text = text.replace(k, v)
        return text

    @staticmethod
    def sent_to_format(sentence: Sentence) -> List[str]:
        outputs: List[str] = []
        for mrph in sentence.morphemes:
            if mrph.surf == "\u3000":
                outputs.extend(
                    [
                        SURF_TOKEN,
                        FULL_SPACE_TOKEN,
                        READING_TOKEN,
                        FULL_SPACE_TOKEN,
                        LEMMA_TOKEN,
                        FULL_SPACE_TOKEN,
                        CANON_TOKEN,
                        "/",
                    ]
                )
            elif mrph.surf == " ":
                outputs.extend(
                    [
                        SURF_TOKEN,
                        HALF_SPACE_TOKEN,
                        READING_TOKEN,
                        HALF_SPACE_TOKEN,
                        LEMMA_TOKEN,
                        HALF_SPACE_TOKEN,
                        CANON_TOKEN,
                        "/",
                    ]
                )
            elif mrph.surf == "…":
                outputs.extend(
                    [
                        SURF_TOKEN,
                        TRIPLE_DOT_TOKEN,
                        READING_TOKEN,
                        TRIPLE_DOT_TOKEN,
                        LEMMA_TOKEN,
                        TRIPLE_DOT_TOKEN,
                        CANON_TOKEN,
                        f"{TRIPLE_DOT_TOKEN}/{TRIPLE_DOT_TOKEN}",
                    ]
                )
            else:
                if mrph.reading == "\u3000":
                    reading: str = FULL_SPACE_TOKEN
                elif "/" in mrph.reading and len(mrph.reading) > 1:
                    reading = mrph.reading.split("/")[0]
                else:
                    reading = mrph.reading
                lemma: str = FULL_SPACE_TOKEN if mrph.lemma == "\u3000" else mrph.lemma
                canon: str = mrph.canon if mrph.canon is not None else NO_CANON_TOKEN
                outputs.extend([SURF_TOKEN, mrph.surf, READING_TOKEN, reading, LEMMA_TOKEN, lemma, CANON_TOKEN, canon])
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

                reading: str = line.split(READING_TOKEN)[1].split(LEMMA_TOKEN)[0]
                reading = self.token_to_word[reading] if reading in self.token_to_word else reading

                lemma: str = line.split(LEMMA_TOKEN)[1].split(CANON_TOKEN)[0]
                lemma = self.token_to_word[lemma] if lemma in self.token_to_word else lemma

                canon: str = line.split(CANON_TOKEN)[1]
                canon = (
                    f"{self.token_to_word[TRIPLE_DOT_TOKEN]}/{self.token_to_word[TRIPLE_DOT_TOKEN]}"
                    if canon == f"{TRIPLE_DOT_TOKEN}/{TRIPLE_DOT_TOKEN}"
                    else canon
                )
                canon = f'"代表表記:{canon}"' if canon != NO_CANON_TOKEN else "NIL"

                formatted += f"{surf} {reading} {lemma} 未定義語 15 その他 1 * 0 * 0 {canon}\n"
            except IndexError:
                formatted += "@ @ @ 未定義語 15 その他 1 * 0 * 0 NIL\n"
        formatted += "EOS\n"
        return Sentence.from_jumanpp(formatted)
