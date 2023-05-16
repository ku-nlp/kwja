import copy
from typing import List

from rhoknp import Sentence

from kwja.utils.constants import FULL_SPACE_TOKEN, NO_CANON_TOKEN


def get_seq2seq_format(sentence: Sentence) -> str:
    output: str = ""
    for mrph in sentence.morphemes:
        if mrph.surf == "\u3000":
            mrph_info: str = f"{FULL_SPACE_TOKEN} {FULL_SPACE_TOKEN} {FULL_SPACE_TOKEN} {NO_CANON_TOKEN}\n"
        else:
            if mrph.reading == "\u3000":
                reading: str = FULL_SPACE_TOKEN
            elif "/" in mrph.reading and len(mrph.reading) > 1:
                reading = mrph.reading.split("/")[0]
            else:
                reading = mrph.reading
            canon: str = mrph.canon if mrph.canon is not None else NO_CANON_TOKEN
            mrph_info = f"{mrph.surf} {reading} {mrph.lemma} {canon}\n"
        output += mrph_info
    return output


def get_sent_from_seq2seq_format(input_text: str) -> Sentence:
    lines: List[str] = input_text.split("\n")
    mrph_placeholder: List[str] = ["@", "@", "@", "未定義語", "15", "その他", "1", "*", "0", "*", "0", "NIL"]
    formatted: str = ""
    for line in lines:
        if not line:
            continue
        if line == "EOS":
            formatted += line + "\n"
        else:
            preds: List[str] = line.split(" ")  # surf reading lemma canon
            mrphs: List[str] = copy.deepcopy(mrph_placeholder)
            if len(preds) == 4:
                mrphs[0] = "\u3000" if preds[0] == FULL_SPACE_TOKEN else preds[0]
                mrphs[1] = "\u3000" if preds[1] == FULL_SPACE_TOKEN else preds[1]
                mrphs[2] = "\u3000" if preds[2] == FULL_SPACE_TOKEN else preds[2]
                if preds[3] == NO_CANON_TOKEN:
                    if preds[0] == FULL_SPACE_TOKEN and preds[1] == FULL_SPACE_TOKEN and preds[2] == FULL_SPACE_TOKEN:
                        mrphs[-1] = '"代表表記:/"'
                    else:
                        mrphs[-1] = "NIL"
                else:
                    mrphs[-1] = f'"代表表記:{preds[3]}"'
            elif line in ["!!!!/!", "????/?", ",,,,/,", "..../."]:
                for idx in range(3):
                    mrphs[idx] = line[idx]
                mrphs[-1] = f'"代表表記:{line[0]}/{line[0]}"'
            elif line == "............/...":
                for idx in range(3):
                    mrphs[idx] = "…"
                mrphs[-1] = '"代表表記:…/…"'
            formatted += " ".join(mrphs) + "\n"
    return Sentence.from_jumanpp(formatted)
