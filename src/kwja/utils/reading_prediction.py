import copy
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import as_file
from typing import Optional, Union

import numpy as np
from rhoknp import Morpheme
from transformers import PreTrainedTokenizerBase

from kwja.utils.constants import (
    CHOON_SET,
    HATSUON_SET,
    ID,
    ID_ID,
    KATA2HIRA,
    LOWER2UPPER,
    PROLONGED_MAP,
    PROLONGED_MAP_FOR_EROW,
    RESOURCE_TRAVERSABLE,
    UNK,
    UNK_ID,
    VOICED2VOICELESS,
)
from kwja.utils.kanjidic import KanjiDic

logger = logging.getLogger(__name__)

READING_VOCAB_TRAVERSABLE = RESOURCE_TRAVERSABLE / "reading_prediction" / "vocab.txt"


def get_reading2reading_id() -> dict[str, int]:
    reading2reading_id = {UNK: UNK_ID, ID: ID_ID}
    with as_file(READING_VOCAB_TRAVERSABLE) as path:
        with open(path) as f:
            for line in f:
                if line := line.strip():
                    if line not in reading2reading_id:
                        reading2reading_id[line] = len(reading2reading_id)
    return reading2reading_id


class ReadingAligner:
    kana_re = re.compile("^[\u3041-\u30ff]+$")

    def __init__(self, tokenizer: PreTrainedTokenizerBase, kanji_dic: KanjiDic) -> None:
        self.tokenizer = tokenizer
        self.kanji_dic = kanji_dic

    def align(self, morphemes: list[Morpheme]) -> list[str]:
        # assumption: morphemes are never combined
        tokenizer_input: Union[list[str], str] = [m.text for m in morphemes]
        encoding = self.tokenizer(tokenizer_input, add_special_tokens=False, is_split_into_words=True).encodings[0]
        word_id2subwords = defaultdict(list)
        for token_id, word_id in enumerate(encoding.word_ids):
            word_id2subwords[word_id].append(self.tokenizer.decode(encoding.ids[token_id]))
        subwords_per_morpheme = [subwords for subwords in word_id2subwords.values()]
        assert len(subwords_per_morpheme) == len(morphemes), (
            f"inconsistent segmentation: {subwords_per_morpheme} / {morphemes}"
        )

        readings: list[str] = []
        for morpheme, subwords in zip(morphemes, subwords_per_morpheme):
            readings.extend(self._align_morpheme(morpheme, subwords))
        return readings

    def _align_morpheme(self, morpheme: Morpheme, subwords: list[str]) -> list[str]:
        # trivial
        if len(subwords) == 1:
            return [morpheme.reading]

        # initial subword can be empty
        if subwords[0] == "":
            subwords = subwords[1:]
            empty_initial = True
        else:
            empty_initial = False

        # surf = morpheme.surf
        reading = morpheme.reading
        boundaries = [0]
        pos = 0
        for subword in subwords:
            pos += len(subword)
            boundaries.append(pos)
        surf = "".join(subwords)
        if surf != morpheme.surf:
            logger.warning(f"non-identical surf forms: {morpheme.surf}\t{surf}")

        @dataclass
        class Node:
            i: int
            j: int
            wI: int
            wJ: int
            cost: int

        # build lattice
        # no node can cross boundaries
        td_lattice: list[list[list[Node]]] = []
        td_holder: list[list[tuple[Optional[Node], Optional[Node], float]]] = []
        node: Optional[Node] = None
        node_prev: Optional[Node]
        for _ in range(len(surf)):
            td_lattice.append([])
            for _ in range(len(reading) + 1):  # +1 for zero-width reading
                td_lattice[-1].append([])
        for _ in range(len(surf) + 1):
            td_holder.append([])
            for _ in range(len(reading) + 1):
                td_holder[-1].append((None, None, np.inf))
        td_holder[0][0] = (None, None, 0)

        for i in range(len(surf)):
            ci = surf[i]
            kanji_reading_list = []
            if ci in self.kanji_dic.entries:
                kanji_reading_list = self._extend_kanji_reading_list(self.kanji_dic.entries[ci]["reading"])
            elif i > 0 and ci == "々":
                if surf[i - 1] in self.kanji_dic.entries:
                    kanji_reading_list = self._extend_kanji_reading_list(self.kanji_dic.entries[surf[i - 1]]["reading"])
            for j in range(len(reading)):
                cj = reading[j]
                if ci.translate(KATA2HIRA) == cj:
                    node = Node(i=i, j=j, wI=1, wJ=1, cost=0)
                    td_lattice[i][j].append(node)
                if ci in VOICED2VOICELESS and cj == VOICED2VOICELESS[ci]:
                    node = Node(i=i, j=j, wI=1, wJ=1, cost=1)
                    td_lattice[i][j].append(node)
                if ci in LOWER2UPPER and cj == LOWER2UPPER[ci]:
                    node = Node(i=i, j=j, wI=1, wJ=1, cost=1)
                    td_lattice[i][j].append(node)
                if ci in CHOON_SET and i > 0:
                    p = surf[i - 1]
                    if p in PROLONGED_MAP and cj == PROLONGED_MAP[p]:
                        node = Node(i=i, j=j, wI=1, wJ=1, cost=2)
                        td_lattice[i][j].append(node)
                    if p in PROLONGED_MAP_FOR_EROW and cj == PROLONGED_MAP_FOR_EROW[p]:
                        node = Node(i=i, j=j, wI=1, wJ=1, cost=2)
                        td_lattice[i][j].append(node)
                for kanji_reading in kanji_reading_list:
                    # if "." in kanji_reading:
                    #     kanji_reading = kanji_reading.split(".")[0]
                    if j + len(kanji_reading) > len(reading):
                        continue
                    reading_part = reading[j : j + len(kanji_reading)]
                    if reading_part == kanji_reading:
                        node = Node(i=i, j=j, wI=1, wJ=len(kanji_reading), cost=0)
                        td_lattice[i][j].append(node)
                    else:
                        # loose matching
                        if reading_part[0] in VOICED2VOICELESS:
                            reading_part2 = VOICED2VOICELESS[reading_part[0]] + reading_part[1:]
                            if reading_part2 == kanji_reading:
                                node = Node(i=i, j=j, wI=1, wJ=len(kanji_reading), cost=10)
                                td_lattice[i][j].append(node)
                        if kanji_reading[-1] in ("き", "く", "ち", "つ"):
                            kanji_reading2 = kanji_reading[:-1] + "っ"
                            if reading_part == kanji_reading2:
                                node = Node(i=i, j=j, wI=1, wJ=len(kanji_reading), cost=10)
                                td_lattice[i][j].append(node)
                        # TODO: combination
                # fallback nodes
                if cj in ("ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゃ", "ゅ", "ょ", "っ", "ん", "ー"):
                    initial_penalty = 500
                else:
                    initial_penalty = 0
                for j2 in range(j + 1, len(reading) + 1):
                    wJ = j2 - j
                    node = Node(i=i, j=j, wI=1, wJ=wJ, cost=initial_penalty + 1000 * int(wJ**1.5))
                    td_lattice[i][j].append(node)
                basenode_list = copy.copy(td_lattice[i][j])  # shallow
                basenode_list2 = []
                for i2 in range(i + 1, len(surf)):
                    # cannot cross boundaries
                    if i2 in boundaries:
                        break
                    ci2 = surf[i2]
                    if ci2 in CHOON_SET or ci2 in HATSUON_SET or ci2 in LOWER2UPPER:
                        for _ in basenode_list:
                            assert isinstance(node, Node)
                            node2 = copy.deepcopy(node)
                            node2.cost += 10
                            node2.wI += 1
                            td_lattice[i][j].append(node2)
                            basenode_list2.append(node2)
                        basenode_list = basenode_list2
                        basenode_list2 = []
                    else:
                        break
            # fallback node 2: zero reading for special signs
            if ci in CHOON_SET or ci in HATSUON_SET or ci in LOWER2UPPER:
                for j in range(len(reading) + 1):
                    node = Node(i=i, j=j, wI=1, wJ=0, cost=500)
                    td_lattice[i][j].append(node)
            elif unicodedata.category(ci)[0] != "L":
                for j in range(len(reading) + 1):
                    node = Node(i=i, j=j, wI=1, wJ=0, cost=500)
                    td_lattice[i][j].append(node)
            else:
                for j in range(len(reading) + 1):
                    node = Node(i=i, j=j, wI=1, wJ=0, cost=5000)
                    td_lattice[i][j].append(node)
        # 2d viterbi
        for i in range(1, len(surf) + 1):
            for j in range(1, len(reading) + 1):
                cands = []
                for i2 in range(0, i):
                    for j2 in range(0, j + 1):  # +1 for zero-width reading
                        node1, _, cost1 = td_holder[i2][j2]
                        for node2 in td_lattice[i2][j2]:
                            if i2 + node2.wI == i and j2 + node2.wJ == j:
                                cands.append((node2, node1, cost1 + node2.cost))
                if len(cands) > 0:
                    td_holder[i][j] = sorted(cands, key=lambda x: x[2])[0]
                else:
                    td_holder[i][j] = (None, None, np.inf)
        # backtracking
        node, node_prev, _ = td_holder[-1][-1]
        seg = []
        while True:
            assert isinstance(node, Node)
            seg.append((node.wI, node.wJ))
            if node_prev is None:
                break
            node, node_prev, _ = td_holder[node.i][node.j]
        seg.reverse()
        if td_holder[-1][-1][2] >= 1000:
            logger.warning(f"{seg}\t{reading}\t{subwords}")

        # dummy
        posI, posJ = 0, 0
        subreadings = []
        if empty_initial:
            subreadings.append("")
        subreading = ""
        boundaries.pop(0)
        while True:
            wI, wJ = seg.pop(0)
            subreading += reading[posJ : posJ + wJ]
            posI += wI
            posJ += wJ
            if posI == boundaries[0]:
                subreadings.append(subreading)
                if len(boundaries) > 1:
                    subreading = ""
                    boundaries.pop(0)
                else:
                    break
        return subreadings

    @staticmethod
    def _extend_kanji_reading_list(kanji_reading_list_orig: list[str]) -> list[str]:
        kanji_reading_list = []
        for kanji_reading in kanji_reading_list_orig:
            kanji_reading = re.sub("-", "", kanji_reading)  # noqa: PLW2901
            if "." in kanji_reading:
                base, ending = kanji_reading.split(".")
                if base not in kanji_reading_list:
                    kanji_reading_list.append(base)
                for i in range(1, len(ending)):
                    kanji_reading2 = base + ending[0:i]
                    if kanji_reading2 not in kanji_reading_list:
                        kanji_reading_list.append(kanji_reading2)
            elif kanji_reading not in kanji_reading_list:
                kanji_reading_list.append(kanji_reading)
        return kanji_reading_list


def get_word_level_readings(readings: list[str], tokens: list[str], subword_map: list[list[bool]]) -> list[str]:
    """サブワードレベルの読みを単語レベルの読みに変換．

    Args:
        readings: list[str] サブワードレベルの読み．
        tokens: list[str] サブワードレベルの表層．
        subword_map: list[list[bool]] subword_map[i][j] = True ならば i 番目の単語は j 番目のトークンを含む．
    """
    ret: list[str] = []
    for flags in subword_map:
        item = ""
        for token, reading, flag in zip(tokens, readings, flags):
            if flag:
                if reading in {UNK, ID}:
                    item += token
                else:
                    item += reading
        if item:
            ret.append(item)
        elif any(flags):
            ret.append("_")
    return ret
