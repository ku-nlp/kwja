import logging

import numpy as np
from jinf import Jinf
from rhoknp import Morpheme, Sentence

from kwja.utils.constants import (
    CHOON_SET,
    HATSUON_SET,
    IGNORE_WORD_NORM_OP_TAG,
    LOWER2UPPER,
    PROLONGED_MAP,
    PROLONGED_MAP_FOR_EROW,
    UPPER2LOWER,
    VOICED2VOICELESS,
)

logger = logging.getLogger(__name__)


class MorphemeNormalizer:
    def __init__(self) -> None:
        self.jinf = Jinf()

    def get_word_norm_op_tags(self, morpheme: Morpheme) -> list[str]:
        try:
            if morpheme.conjtype == "*":
                norm = morpheme.lemma
            else:
                norm = self.jinf(morpheme.lemma, morpheme.conjtype, "基本形", morpheme.conjform)
            return get_word_norm_op_tags(morpheme.surf, norm)
        except ValueError as e:
            logger.info(f"failed to get normalized form of {morpheme.surf}: {e}")
            return [IGNORE_WORD_NORM_OP_TAG] * len(morpheme.surf)


def get_word_norm_op_tags(surf: str, norm: str) -> list[str]:
    m, n = len(surf) + 1, len(norm) + 1
    if m < n:
        raise ValueError(f"failed to get word norm op tags from {surf} and {norm}")
    dp: list[list[tuple[float, str]]] = [[(float("inf"), "_") for _ in range(n)] for _ in range(m)]  # (m, n)
    dp[0][0] = (0.0, "")
    for j in range(1, n):
        cj = norm[j - 1]
        for i in range(1, m):
            if i < j:
                continue
            ci = surf[i - 1]
            candidates = []
            # heuristic: add D before K
            if ci in CHOON_SET or ci in HATSUON_SET or ci in LOWER2UPPER:
                candidates.append((dp[i - 1][j][0] + 1.0, "D"))
            if ci == cj:
                candidates.append((dp[i - 1][j - 1][0], "K"))
            else:
                if i == 1 and j == 1 and ci in VOICED2VOICELESS and cj == VOICED2VOICELESS[ci]:
                    candidates.append((dp[i - 1][j - 1][0] + 1.0, "V"))
                if ci in LOWER2UPPER and cj == LOWER2UPPER[ci]:
                    candidates.append((dp[i - 1][j - 1][0] + 1.0, "S"))
                if ci in CHOON_SET and i >= 2 and dp[i - 1][j - 1][1] == "K":
                    # NOTE: "P" and "E" must follow "K"
                    # Juman++ does not support いくぇー -> いくえい
                    p = surf[i - 2]
                    if p in PROLONGED_MAP and cj == PROLONGED_MAP[p]:
                        candidates.append((dp[i - 1][j - 1][0] + 1.0, "P"))
                    if p in PROLONGED_MAP_FOR_EROW and cj == PROLONGED_MAP_FOR_EROW[p]:
                        candidates.append((dp[i - 1][j - 1][0] + 1.0, "E"))
            if len(candidates) >= 1:
                dp[i][j] = sorted(candidates)[0]  # choose the least-cost word norm op tag
    if dp[-1][-1] == (float("inf"), "_"):
        raise ValueError(f"failed to get word norm op tags from {surf} and {norm}")
    # backtracking
    i, j = m - 1, n - 1
    word_norm_op_tags = []
    while i >= 0 and j >= 0 and not (i == j == 0):
        word_norm_op_tag = dp[i][j][1]
        word_norm_op_tags.append(word_norm_op_tag)
        if word_norm_op_tag == "D":
            i -= 1
        else:
            i -= 1
            j -= 1
    word_norm_op_tags.reverse()
    return word_norm_op_tags


def get_normalized_surf(surf: str, word_norm_op_tags: list[str], strict: bool = True) -> str:
    assert len(surf) == len(word_norm_op_tags)
    norm = ""
    for i, (c, word_norm_op_tag) in enumerate(zip(surf, word_norm_op_tags)):
        if word_norm_op_tag == "K":
            norm += c
        elif word_norm_op_tag == "V":
            if strict is True and i != 0:
                raise ValueError(f"not an initial kana {c} in {surf}")
            if c in VOICED2VOICELESS:
                norm += VOICED2VOICELESS[c]
            else:
                if strict is True:
                    raise ValueError(f"not a voiced kana {c} in {surf}")
                norm += c
        elif word_norm_op_tag == "D":
            if strict is True and c not in CHOON_SET and c not in HATSUON_SET and c not in LOWER2UPPER:
                raise ValueError(f"not a removable kana {c} in {surf}")
        elif word_norm_op_tag == "S":
            if c in LOWER2UPPER:
                norm += LOWER2UPPER[c]
            else:
                if strict is True:
                    raise ValueError(f"not a small kana {c} in {surf}")
                norm += c
        elif word_norm_op_tag == "P":
            # NOTE: in canonical ops, P and E must follow K, but we do not check this constraint here
            if len(norm) <= 0:
                if strict is True:
                    raise ValueError(f"no preceding kana for {c} in {surf}")
                norm += c
            elif c not in CHOON_SET:
                if strict is True:
                    raise ValueError(f"not a prolonged sign for {c} in {surf}")
                norm += c
            else:
                p = norm[-1]
                if p in PROLONGED_MAP:
                    norm += PROLONGED_MAP[p]
                else:
                    if strict is True:
                        raise ValueError(f"not a valid preceding kana {p} in {surf}")
                    norm += c
        elif word_norm_op_tag == "E":
            if len(norm) <= 0:
                if strict is True:
                    raise ValueError(f"no preceding kana for {c} in {surf}")
                norm += c
            elif c not in CHOON_SET:
                if strict is True:
                    raise ValueError(f"not a prolonged sign for {c} in {surf}")
                norm += c
            else:
                p = norm[-1]
                if p in PROLONGED_MAP_FOR_EROW:
                    norm += PROLONGED_MAP_FOR_EROW[p]
                else:
                    if strict is True:
                        raise ValueError(f"not a valid preceding kana {p} in {surf}")
                    norm += c
        else:
            raise NotImplementedError(f"unknown word norm op tag {word_norm_op_tag}")
    return norm


class SentenceDenormalizer:
    def __init__(self) -> None:
        self.mn = MorphemeNormalizer()
        self.md = MorphemeDenormalizer()
        self.rng = np.random.default_rng()

    def denormalize(self, sentence: Sentence, p: float = 0.5) -> None:
        prob = p
        for morpheme in reversed(sentence.morphemes):
            if not self._is_normal_morpheme(morpheme):
                continue
            surf2 = self.md.denormalize(morpheme)
            if surf2 != morpheme.text:
                if self.rng.random() < prob:
                    morpheme.text = surf2
                    prob *= 0.1
            else:
                prob = min(prob * 1.5, p)

    def _is_normal_morpheme(self, morpheme: Morpheme) -> bool:
        opn = self.mn.get_word_norm_op_tags(morpheme)
        if opn[0] == IGNORE_WORD_NORM_OP_TAG:
            return False
        return all(map(lambda x: x in ("K",), opn))


class MorphemeDenormalizer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def denormalize(self, morpheme: Morpheme) -> str:
        return self._denormalize(morpheme.surf)

    def _denormalize(self, surf: str) -> str:
        if len(surf) == 1:
            return surf
        surf2 = self._denormalize_deterministic(surf, var_prolonged=True)
        cands: list[tuple[str, float]] = [(surf, 10.0)]
        # D: ー, っ
        if is_kana(surf2[-1]):
            d = 0.1
            if surf2[-1] in ("ん", "ン", "っ", "ッ"):
                d *= 0.1
            cands.append((surf2 + "ー", d * 1.0))
            cands.append((surf2 + "ーー", d * 0.1))
            cands.append((surf2 + "ーーー", d * 0.05))
            cands.append((surf2 + "〜", d * 0.2))
            cands.append((surf2 + "〜〜", d * 0.02))
            cands.append((surf2 + "〜〜〜", d * 0.001))
            cands.append((surf2 + "っ", d * 0.1))
            cands.append((surf2 + "ッ", d * 0.05))
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
            cands.append((surf2 + "ぁ", d * 1.0))
            cands.append((surf2 + "ぁー", d * 0.1))
            cands.append((surf2 + "ぁーー", d * 0.05))
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
            cands.append((surf2 + "ぅ", d * 1.0))
            cands.append((surf2 + "ぅー", d * 0.01))
            cands.append((surf2 + "ぅーー", d * 0.05))
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
            cands.append((surf2 + "ぉ", d * 1.0))
            cands.append((surf2 + "ぉー", d * 0.1))
            cands.append((surf2 + "ぉーー", d * 0.05))
        elif surf2[-1] == "っ":
            d = 0.5
            cands.append((surf2 + "っ", d * 1.0))
            cands.append((surf2 + "っっ", d * 0.1))
            cands.append((surf2 + "っっ", d * 0.05))
        elif len(surf2) >= 2 and surf2[-1] == "い" and is_kana(surf2[-2]):
            d = 1.0
            cands.append((surf2[:-1] + "ーい", d * 1.0))
            cands.append((surf2[:-1] + "ーーい", d * 0.1))
        elif len(surf2) >= 2 and surf2[-1] == "く" and is_kana(surf2[-2]):
            d = 1.0
            cands.append((surf2[:-1] + "ーく", d * 1.0))
            cands.append((surf2[:-1] + "ーーく", d * 0.1))
        if surf2 == "です":
            cands.append(("でーす", 1.0))
            cands.append(("で〜す", 0.2))
        if surf2 == "ます":
            cands.append(("まーす", 1.0))
            cands.append(("ま〜す", 0.2))
        if len(cands) <= 1:
            return surf2
        probs = np.array(list(map(lambda x: x[1], cands)))
        probs /= probs.sum()
        idx = self.rng.choice(len(probs), p=probs)
        return cands[idx][0]

    def _denormalize_deterministic(self, surf: str, var_prolonged: bool = False) -> str:
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
        if var_prolonged and self.rng.random() < 0.1:
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


def is_kana(char: str) -> bool:
    # this ignores extended kana
    if "\u3040" <= char <= "\u30ff":
        return True
    return False


def find_kanji_ga(surf: str) -> int:
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
    if "\u4e00" <= char <= "\u9fff":
        return True
    return False
