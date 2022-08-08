import logging

import numpy as np
from jinf import Jinf
from rhoknp import Morpheme

from jula.utils.constants import IGNORE_CHARNORM_TYPE

logger = logging.getLogger(__name__)

# KATAKANA-HIRAGANA PROLONGED SOUND MARK (0x30fc)
# "〜"(0x301C)  "⁓" (U+2053)、Full-width tilde:
# "～" (U+FF5E)、tilde operator: "∼" (U+223C)
# Half-widths HIRAGANA-KATAKANA PROLONGED SOUND MARK (U+FF70
CHOON_SET = set(["ー", "〜", "～", "∼", "ｰ"])

HATSUON_SET = set(["っ", "ッ"])

LOWER2UPPER = {
    "ぁ": "あ",
    "ぃ": "い",
    "ぅ": "う",
    "ぇ": "え",
    "ぉ": "お",
    "ゎ": "わ",
    "ヶ": "ケ",
    "ケ": "ヶ",
}

VOICED2VOICELESS = {
    "が": "か",
    "ぎ": "き",
    "ぐ": "く",
    "げ": "け",
    "ご": "こ",
    "ガ": "カ",
    "ギ": "キ",
    "グ": "ク",
    "ゲ": "ケ",
    "ゴ": "コ",
    "ざ": "さ",
    "じ": "し",
    "ず": "す",
    "ぜ": "せ",
    "ぞ": "そ",
    "ザ": "サ",
    "ジ": "シ",
    "ズ": "ス",
    "ゼ": "セ",
    "ゾ": "ソ",
    "だ": "た",
    "ぢ": "ち",
    "づ": "つ",
    "で": "て",
    "ど": "と",
    "ダ": "タ",
    "ヂ": "チ",
    "ヅ": "ツ",
    "デ": "テ",
    "ド": "ト",
    "ば": "は",
    "び": "ひ",
    "ぶ": "ふ",
    "べ": "へ",
    "ぼ": "ほ",
    "バ": "ハ",
    "ビ": "ヒ",
    "ブ": "フ",
    "ベ": "ヘ",
    "ボ": "ホ",
    "ぱ": "は",
    "ぴ": "ひ",
    "ぷ": "ふ",
    "ぺ": "へ",
    "ぽ": "ほ",
    "パ": "ハ",
    "ピ": "ヒ",
    "プ": "フ",
    "ペ": "ヘ",
    "ポ": "ホ",
}

PROLONGED_MAP = {
    "か": "あ",
    "が": "あ",
    "ば": "あ",
    "ま": "あ",
    "ゃ": "あ",
    "い": "い",
    "き": "い",
    "し": "い",
    "ち": "い",
    "に": "い",
    "ひ": "い",
    "じ": "い",
    "け": "い",
    "せ": "い",
    "へ": "い",
    "め": "い",
    "れ": "い",
    "げ": "い",
    "ぜ": "い",
    "で": "い",
    "べ": "い",
    "ぺ": "い",
    "く": "う",
    "す": "う",
    "つ": "う",
    "ふ": "う",
    "ゆ": "う",
    "ぐ": "う",
    "ず": "う",
    "ぷ": "う",
    "ゅ": "う",
    "お": "う",
    "こ": "う",
    "そ": "う",
    "と": "う",
    "の": "う",
    "ほ": "う",
    "も": "う",
    "よ": "う",
    "ろ": "う",
    "ご": "う",
    "ぞ": "う",
    "ど": "う",
    "ぼ": "う",
    "ぽ": "う",
    "ょ": "う",
    "え": "い",
    "ね": "い",
}

PROLONGED_MAP_FOR_EROW = {
    "え": "え",
    "け": "え",
    "げ": "え",
    "せ": "え",
    "ぜ": "え",
    "て": "え",
    "で": "え",
    "ね": "え",
    "へ": "え",
    "べ": "え",
    "め": "え",
    "れ": "え",
}


class MorphemeNormalizer:
    def __init__(self) -> None:
        self.jinf = Jinf()

    def get_normalization_opns(self, morpheme: Morpheme) -> list[str]:
        try:
            if morpheme.conjtype == "*":
                normalized = morpheme.surf
            else:
                normalized = self.jinf(morpheme.lemma, morpheme.conjtype, "基本形", morpheme.conjform)
            return get_normalization_opns(morpheme.surf, normalized)
        except ValueError as e:
            logger.info(f"failed to get normalized form of {morpheme.surf}: {e}")
            return [IGNORE_CHARNORM_TYPE] * len(morpheme.surf)


def get_normalization_opns(surf: str, normalized: str) -> list[str]:
    surf_len = len(surf) + 1
    normalized_len = len(normalized) + 1
    if surf_len < normalized_len:
        raise ValueError(f"failed to construct normalization labels to convert {surf} to {normalized}")
    d = np.inf * np.ones((surf_len, normalized_len), dtype=np.int32)
    dops: list[list[str]] = []
    for i in range(surf_len):
        dops.append([])
        for j in range(normalized_len):
            dops[-1].append("_")
    d[0, 0] = 0
    dops[0][0] = ""
    for j in range(1, normalized_len):
        cj = normalized[j - 1]
        for i in range(1, surf_len):
            if i < j:
                continue
            ci = surf[i - 1]
            lops = []
            costs = []
            # heuristics: add D before K
            if ci in CHOON_SET or ci in HATSUON_SET or ci in LOWER2UPPER:
                lops.append("D")
                costs.append(d[i - 1, j] + 1)
            if ci == cj:
                lops.append("K")
                costs.append(d[i - 1, j - 1])
            else:
                if i == 1 and j == 1 and ci in VOICED2VOICELESS and cj == VOICED2VOICELESS[ci]:
                    lops.append("V")
                    costs.append(d[i - 1, j - 1] + 1)
                if ci in LOWER2UPPER and cj == LOWER2UPPER[ci]:
                    lops.append("S")
                    costs.append(d[i - 1, j - 1] + 1)
                if ci in CHOON_SET and i > 1 and dops[i - 1][j - 1] == "K":
                    # NOTE: "P" and "E" must follow "K"
                    # jumanpp does not support いくぇー -> いくえい
                    p = surf[i - 2]
                    if p in PROLONGED_MAP and cj == PROLONGED_MAP[p]:
                        lops.append("P")
                        costs.append(d[i - 1, j - 1] + 1)
                    if p in PROLONGED_MAP_FOR_EROW and cj == PROLONGED_MAP_FOR_EROW[p]:
                        lops.append("E")
                        costs.append(d[i - 1, j - 1] + 1)
            if len(lops) > 0:
                idx = np.array(costs).argmin()
                d[i, j] = costs[idx]
                dops[i][j] = lops[idx]
    if np.isinf(d[-1, -1]):
        raise ValueError(f"failed to construct normalization labels to convert {surf} to {normalized}")
    # backtracking
    i, j = surf_len - 1, normalized_len - 1
    ops = []
    while i >= 0 and j >= 0 and not (i == j == 0):
        op = dops[i][j]
        assert op not in ("", "_")
        ops.append(op)
        if op == "D":
            i -= 1
        else:
            i -= 1
            j -= 1
    assert len(ops) == surf_len - 1
    ops.reverse()
    return ops


def get_normalized(surf: str, ops: list[str], strict: bool = True) -> str:
    assert len(surf) == len(ops)
    normalized = ""
    for i, (c, op) in enumerate(zip(surf, ops)):
        if op == "K":
            normalized += c
        elif op == "V":
            if strict and i != 0:
                raise ValueError(f"not an initial kana {c} in {surf}")
            if c in VOICED2VOICELESS:
                normalized += VOICED2VOICELESS[c]
            else:
                if strict:
                    raise ValueError(f"not a voiced kana {c} in {surf}")
                normalized += c
        elif op == "D":
            if strict and c not in CHOON_SET and c not in HATSUON_SET and c not in LOWER2UPPER:
                raise ValueError(f"not a removable kana {c} in {surf}")
        elif op == "S":
            if c in LOWER2UPPER:
                normalized += LOWER2UPPER[c]
            else:
                if strict:
                    raise ValueError(f"not a small kana {c} in {surf}")
                normalized += c
        elif op == "P":
            # NOTE: in cannonical ops, P and E must follow K
            # but we do not check this constraint here
            if len(normalized) <= 0:
                if strict:
                    raise ValueError(f"no preceding kana for {c} in {surf}")
                normalized += c
            elif c not in CHOON_SET:
                if strict:
                    raise ValueError(f"not a prolonged sign for {c} in {surf}")
                normalized += c
            else:
                p = normalized[-1]
                if p in PROLONGED_MAP:
                    normalized += PROLONGED_MAP[p]
                else:
                    if strict:
                        raise ValueError(f"not a valid precding kana {p} in {surf}")
                    normalized += c
        elif op == "E":
            if len(normalized) <= 0:
                if strict:
                    raise ValueError(f"no preceding kana for {c} in {surf}")
                normalized += c
            elif c not in CHOON_SET:
                if strict:
                    raise ValueError(f"not a prolonged sign for {c} in {surf}")
                normalized += c
            else:
                p = normalized[-1]
                if p in PROLONGED_MAP_FOR_EROW:
                    normalized += PROLONGED_MAP_FOR_EROW[p]
                else:
                    if strict:
                        raise ValueError(f"not a valid precding kana {p} in {surf}")
                    normalized += c
        else:
            raise NotImplementedError(f"unknown op {op}")
    return normalized


if __name__ == "__main__":
    # print(get_normalized("がえるー", ["V", "K", "K", "D"], strict=True) == "かえる")
    print(get_normalization_opns("あーーーー", "あー"))
    print(get_normalization_opns("ね〜", "ねえ"))
