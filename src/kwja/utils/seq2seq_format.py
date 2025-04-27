from rhoknp import Sentence
from transformers import PreTrainedTokenizerFast

from kwja.utils.constants import (
    CANON_TOKEN,
    HALF_SPACE_TOKEN,
    LEMMA_TOKEN,
    MORPHEME_DELIMITER_TOKEN,
    NO_CANON_TOKEN,
    RARE2SPECIAL,
    READING_TOKEN,
    SPECIAL2RARE,
    SURF_TOKEN,
)


class Seq2SeqFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer: PreTrainedTokenizerFast = tokenizer

        self.word2token: dict[str, str] = {" ": HALF_SPACE_TOKEN}
        self.token2word: dict[str, str] = {v: k for k, v in self.word2token.items()}

    def get_surfs(self, sentence: Sentence) -> list[str]:
        surfs: list[str] = []
        for morpheme in sentence.morphemes:
            surf: str = morpheme.surf
            for word, token in self.word2token.items():
                surf = surf.replace(word, token)
            for k, v in RARE2SPECIAL.items():
                surf = surf.replace(k, v)
            tokenized_surf: list[str] = [x for x in self.tokenizer.tokenize(surf) if x != "▁"]
            decoded: str = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokenized_surf))
            for token in self.word2token.values():
                decoded = decoded.replace(f"{token} ", token)
            for token in SPECIAL2RARE:
                decoded = decoded.replace(f"{token} ", token)
            surfs.append(decoded.replace(" ", HALF_SPACE_TOKEN))
        return surfs

    def get_src_tokens(self, sentence: Sentence) -> list[str]:
        src_text: str = MORPHEME_DELIMITER_TOKEN.join(m.surf for m in sentence.morphemes)
        for word, token in self.word2token.items():
            src_text = src_text.replace(word, token)
        for rare, special in RARE2SPECIAL.items():
            src_text = src_text.replace(rare, special)
        return [x for x in self.tokenizer.tokenize(src_text) if x != "▁"]

    def get_tgt_tokens(self, sentence: Sentence) -> list[str]:
        seq2seq_format: str = ""
        for morpheme in sentence.morphemes:
            if morpheme.surf == " ":
                surf = HALF_SPACE_TOKEN
                reading = HALF_SPACE_TOKEN
                lemma = HALF_SPACE_TOKEN
                canon = "/"
            else:
                surf = morpheme.surf
                if morpheme.reading == " ":  # 読みがない場合
                    reading = HALF_SPACE_TOKEN
                elif "/" in morpheme.reading and len(morpheme.reading) > 1:
                    reading = morpheme.reading.split("/")[0]
                else:
                    reading = morpheme.reading
                lemma = morpheme.lemma
                if morpheme.canon is not None:
                    canon = morpheme.canon
                    canon_list: list[str] = canon.split("/")
                    if len(canon_list) > 2 and canon_list[0] and canon_list[1]:
                        canon = f"{canon_list[0]}/{canon_list[1]}"
                else:
                    canon = NO_CANON_TOKEN
            seq2seq_format += f"{SURF_TOKEN}{surf}{READING_TOKEN}{reading}{LEMMA_TOKEN}{lemma}{CANON_TOKEN}{canon}"
        for k, v in RARE2SPECIAL.items():
            seq2seq_format = seq2seq_format.replace(k, v)
        return [x for x in self.tokenizer.tokenize(seq2seq_format) if x != "▁"]

    def format_to_sent(self, text: str) -> Sentence:
        formatted: str = ""
        for line in text.split(SURF_TOKEN):
            if not line:
                continue
            try:
                surf: str = line.split(READING_TOKEN)[0]
                surf = self.token2word[surf] if surf in self.token2word else surf
                for k, v in SPECIAL2RARE.items():
                    surf = surf.replace(k, v)

                reading: str = line.split(READING_TOKEN)[1].split(LEMMA_TOKEN)[0]
                reading = self.token2word[reading] if reading in self.token2word else reading
                for k, v in SPECIAL2RARE.items():
                    reading = reading.replace(k, v)

                lemma: str = line.split(LEMMA_TOKEN)[1].split(CANON_TOKEN)[0]
                lemma = self.token2word[lemma] if lemma in self.token2word else lemma
                for k, v in SPECIAL2RARE.items():
                    lemma = lemma.replace(k, v)

                canon: str = line.split(CANON_TOKEN)[1]
                for k, v in self.token2word.items():
                    canon = canon.replace(k, v)
                for k, v in SPECIAL2RARE.items():
                    canon = canon.replace(k, v)
                canon = f'"代表表記:{canon}"' if canon != NO_CANON_TOKEN else "NIL"

                # 例外処理
                if surf == " ":
                    reading = " "
                    lemma = " "
                    canon = r'"代表表記:\␣/\␣"'

                formatted += f"{surf} {reading} {lemma} 未定義語 15 その他 1 * 0 * 0 {canon}\n"
            except IndexError:
                formatted += "@ @ @ 未定義語 15 その他 1 * 0 * 0 NIL\n"
        formatted += "EOS\n"
        return Sentence.from_jumanpp(formatted)
