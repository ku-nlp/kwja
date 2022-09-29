import logging

import numpy as np
from jinf import Jinf
from rhoknp import Morpheme, Sentence

from kwja.utils.constants import IGNORE_WORD_NORM_TYPE

logger = logging.getLogger(__name__)

# full-width tilde: "～" (U+FF5E), "〜"(0x301C)
# tilde operator: "∼" (U+223C), "⁓" (U+2053), "~" (U+007E)
# KATAKANA-HIRAGANA PROLONGED SOUND MARK: "ー" (0x30FC)
# half-widths HIRAGANA-KATAKANA PROLONGED SOUND MARK: "ｰ" (U+FF70), "-" (U+002D)
CHOON_SET = {"～", "〜", "∼", "⁓", "~", "ー", "ｰ", "-"}

HATSUON_SET = {"っ", "ッ"}

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
            return [IGNORE_WORD_NORM_TYPE] * len(morpheme.surf)


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
            # NOTE: in canonical ops, P and E must follow K
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
                        raise ValueError(f"not a valid preceding kana {p} in {surf}")
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
                        raise ValueError(f"not a valid preceding kana {p} in {surf}")
                    normalized += c
        else:
            raise NotImplementedError(f"unknown op {op}")
    return normalized


UPPER2LOWER = {
    "あ": "ぁ",
    "い": "ぃ",
    "う": "ぅ",
    "え": "ぇ",
    "お": "ぉ",
    "わ": "ゎ",
}


class SentenceDenormalizer:
    def __init__(self):
        self.mn = MorphemeNormalizer()
        self.md = MorphemeDenormalizer()

    def denormalize(self, sentence: Sentence, p=0.5) -> None:
        prob = p
        for morpheme in reversed(sentence.morphemes):
            if not self._is_normal_morpheme(morpheme):
                continue
            surf2 = self.md.denormalize(morpheme)
            if surf2 != morpheme.surf:
                if np.random.rand() < prob:
                    morpheme._attributes.surf = surf2
                    prob *= 0.1
            else:
                prob = min(prob * 1.5, p)

    def _is_normal_morpheme(self, morpheme):
        opn = self.mn.get_normalization_opns(morpheme)
        if opn[0] == IGNORE_WORD_NORM_TYPE:
            return False
        return all(map(lambda x: x in ("K"), opn))


class MorphemeDenormalizer:
    def denormalize(self, morpheme: Morpheme) -> str:
        return self._denormalize(morpheme.surf)

    def _denormalize(self, surf: str) -> str:
        if len(surf) == 1:
            return surf
        surf2 = self._denormalize_deterministic(surf, var_prolonged=True)
        cands = [[surf, 10.0]]
        # D: ー, っ
        if is_kana(surf2[-1]):
            d = 0.1
            if surf2[-1] in ("ん", "ン", "っ", "ッ"):
                d *= 0.1
            cands.append([surf2 + "ー", d * 1.0])
            cands.append([surf2 + "ーー", d * 0.1])
            cands.append([surf2 + "ーーー", d * 0.05])
            cands.append([surf2 + "〜", d * 0.2])
            cands.append([surf2 + "〜〜", d * 0.02])
            cands.append([surf2 + "〜〜〜", d * 0.001])
            cands.append([surf2 + "っ", d * 0.1])
            cands.append([surf2 + "ッ", d * 0.05])
        # D: ぁ, ぅ, ぉ, っ
        if surf2[-1] in (
            "あ",
            "か",
            "さ",
            "た",
            "な",
            "は",
            "ま",
            "や",
            "ら",
            "わ",
            "が",
            "ざ",
            "だ",
            "ば",
            "ぱ",
            "ゃ",
        ):
            d = 0.5
            cands.append([surf2 + "ぁ", d * 1.0])
            cands.append([surf2 + "ぁー", d * 0.1])
            cands.append([surf2 + "ぁーー", d * 0.05])
        elif surf2[-1] in (
            "う",
            "く",
            "す",
            "つ",
            "ぬ",
            "ふ",
            "む",
            "ゆ",
            "る",
            "ぐ",
            "ず",
            "づ",
            "ぶ",
            "ぷ",
            "ゅ",
        ):
            d = 0.5
            cands.append([surf2 + "ぅ", d * 1.0])
            cands.append([surf2 + "ぅー", d * 0.01])
            cands.append([surf2 + "ぅーー", d * 0.05])
        elif surf2[-1] in (
            "お",
            "こ",
            "そ",
            "と",
            "の",
            "ほ",
            "も",
            "よ",
            "ろ",
            "ご",
            "ぞ",
            "ど",
            "ぼ",
            "ぽ",
            "ょ",
        ):
            d = 0.5
            cands.append([surf2 + "ぉ", d * 1.0])
            cands.append([surf2 + "ぉー", d * 0.1])
            cands.append([surf2 + "ぉーー", d * 0.05])
        elif surf2[-1] == "っ":
            d = 0.5
            cands.append([surf2 + "っ", d * 1.0])
            cands.append([surf2 + "っっ", d * 0.1])
            cands.append([surf2 + "っっ", d * 0.05])
        elif len(surf2) >= 2 and surf2[-1] == "い" and is_kana(surf2[-2]):
            d = 1.0
            cands.append([surf2[:-1] + "ーい", d * 1.0])
            cands.append([surf2[:-1] + "ーーい", d * 0.1])
        elif len(surf2) >= 2 and surf2[-1] == "く" and is_kana(surf2[-2]):
            d = 1.0
            cands.append([surf2[:-1] + "ーく", d * 1.0])
            cands.append([surf2[:-1] + "ーーく", d * 0.1])
        if surf2 == "です":
            cands.append(["でーす", 1.0])
            cands.append(["で〜す", 0.2])
        if surf2 == "ます":
            cands.append(["まーす", 1.0])
            cands.append(["ま〜す", 0.2])
        if len(cands) <= 1:
            return surf2
        probs = np.array(list(map(lambda x: x[1], cands)))
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        return cands[idx][0]

    def _denormalize_deterministic(self, surf, var_prolonged=False) -> str:
        # S
        if surf[-1] in UPPER2LOWER:
            surf2 = surf[:-1] + UPPER2LOWER[surf[-1]]
            return surf2
        # S: ケ/ヶ
        pos = find_kanji_ga(surf)
        if pos >= 0:
            c = "ヶ" if surf[pos] == "ケ" else "ケ"
            surf2 = surf[:pos] + c + surf[pos + 1 :]
            return surf2
        # P, E
        if var_prolonged and np.random.rand() < 0.1:
            prolonged = "〜"
        else:
            prolonged = "ー"
        for i, c in reversed(list(enumerate(surf))):
            if i == 0:
                break
            # よう -> よー
            # ずうっと -> ずーっと
            # もうれつ -> もーれつ
            # びみょう -> びみょー
            # もどかしい -> もどかしー
            if c == "う":
                if surf[i - 1] in (
                    "う",
                    "お",
                    "く",
                    "こ",
                    "す",
                    "そ",
                    "つ",
                    "と",
                    "ぬ",
                    "の",
                    "ふ",
                    "ほ",
                    "む",
                    "も",
                    "ゆ",
                    "よ",
                    "る",
                    "ろ",
                    "ぐ",
                    "ご",
                    "ず",
                    "ぞ",
                    "づ",
                    "ど",
                    "ぶ",
                    "ぼ",
                    "ぷ",
                    "ぽ",
                    "ゅ",
                    "ょ",
                ):
                    surf2 = surf[:i] + prolonged + surf[i + 1 :]
                    return surf2
            if c == "い" and surf[i - 1] == "し":
                surf2 = surf[:i] + prolonged + surf[i + 1 :]
                return surf2
            if c == "え":
                if surf[i - 1] in (
                    "け",
                    "せ",
                    "て",
                    "ね",
                    "へ",
                    "め",
                    "れ",
                    "げ",
                    "ぜ",
                    "で",
                    "べ",
                    "ぺ",
                ):
                    surf2 = surf[:i] + prolonged + surf[i + 1 :]
                    return surf2
        # no rule applicable
        return surf


def is_kana(char):
    # this ignores extended kana
    if char >= "\u3040" and char <= "\u30FF":
        return True
    return False


def find_kanji_ga(surf) -> int:
    # "ケ": "ヶ",
    # "ヶ": "ケ",
    pos = surf.find("ケ", 1, len(surf) - 1)
    if pos < 0:
        pos = surf.find("ヶ", 1, len(surf) - 1)
        if pos < 0:
            return -1
    if is_chinese_char(surf[pos - 1]) and is_chinese_char(surf[pos + 1]):
        return pos
    return -1


def is_chinese_char(char: str) -> bool:
    # this ignores minor Chinese characters
    if char >= "\u4E00" and char <= "\u9FFF":
        return True
    return False


if __name__ == "__main__":
    # print(get_normalized("がえるー", ["V", "K", "K", "D"], strict=True) == "かえる")
    # print(get_normalization_opns("あーーーー", "あー"))
    # print(get_normalization_opns("ね〜", "ねえ"))
    print(MorphemeDenormalizer()._denormalize_deterministic("なあ"))
