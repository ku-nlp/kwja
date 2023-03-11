import copy
from typing import List

from rhoknp import Sentence

from kwja.utils.constants import NO_CANON_TOKEN, NO_READING_TOKEN


def get_seq2seq_format(sentence: Sentence) -> str:
    output: str = ""
    for mrph in sentence.morphemes:
        if mrph.reading == "\u3000":
            reading: str = NO_READING_TOKEN
        elif "/" in mrph.reading:
            reading = mrph.reading.split("/")[0]
        else:
            reading = mrph.reading
        canon: str = mrph.canon if mrph.canon is not None else NO_CANON_TOKEN
        mrph_info: str = f"{mrph.surf} {reading} {mrph.lemma} {canon}\n"
        output += mrph_info
    return output


def get_sent_from_seq2seq_format(input_text: str) -> Sentence:
    lines: List[str] = input_text.split("\n")
    mrph_placeholder: List[str] = ["@", "@", "@", "未定義語", "15", "その他", "1", "*", "0", "*", "0", "NIL"]
    formatted: str = ""
    for line in lines:
        if not line:
            continue
        if line == "EOS" or line.startswith("*") or line.startswith("+"):
            formatted += line + "\n"
        else:
            preds: List[str] = line.split(" ")
            if len(preds) == 4:
                mrphs: List[str] = copy.deepcopy(mrph_placeholder)
                for idx in range(3):
                    mrphs[idx] = preds[idx]
                mrphs[-1] = "NIL" if preds[3] == NO_CANON_TOKEN else f'"代表表記:{preds[3]}"'
                formatted += " ".join(mrphs) + "\n"
            elif line in ["!!!!/!", "????/?", ",,,,/,"]:
                mrphs = copy.deepcopy(mrph_placeholder)
                for idx in range(3):
                    mrphs[idx] = line[idx]
                mrphs[-1] = f'"代表表記:{line[-1]}/{line[-1]}"'
                formatted += " ".join(mrphs) + "\n"
            elif line == "............/...":
                mrphs = copy.deepcopy(mrph_placeholder)
                for idx in range(3):
                    mrphs[idx] = "…"
                mrphs[-1] = '"代表表記:…/…"'
                formatted += " ".join(mrphs) + "\n"
            else:
                formatted += " ".join(mrph_placeholder) + "\n"
    return Sentence.from_jumanpp(formatted)
