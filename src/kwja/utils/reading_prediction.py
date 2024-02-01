import copy
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from rhoknp import Document, Morpheme
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.utils.constants import (
    CHOON_SET,
    HATSUON_SET,
    ID,
    ID_ID,
    KATA2HIRA,
    LOWER2UPPER,
    PROLONGED_MAP,
    PROLONGED_MAP_FOR_EROW,
    UNK,
    UNK_ID,
    VOICED2VOICELESS,
)
from kwja.utils.kanjidic import KanjiDic

logger = logging.getLogger(__name__)


def get_reading2reading_id(path: Path) -> Dict[str, int]:
    reading2reading_id = {UNK: UNK_ID, ID: ID_ID}
    with path.open() as f:
        for line in f:
            if line := line.strip():
                if line not in reading2reading_id:
                    reading2reading_id[line] = len(reading2reading_id)
    return reading2reading_id


class ReadingAligner:
    kana_re = re.compile("^[\u3041-\u30FF]+$")

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, tokenizer_input_format: Literal["words", "text"], kanji_dic: KanjiDic
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_input_format = tokenizer_input_format
        self.kanji_dic = kanji_dic

    def align(self, morphemes: List[Morpheme]) -> List[str]:
        # assumption: morphemes are never combined
        tokenizer_input: Union[List[str], str] = [m.text for m in morphemes]
        if self.tokenizer_input_format == "text":
            tokenizer_input = " ".join(tokenizer_input)
        encoding = self.tokenizer(
            tokenizer_input, add_special_tokens=False, is_split_into_words=self.tokenizer_input_format == "words"
        ).encodings[0]
        word_id2subwords = defaultdict(list)
        for token_id, word_id in enumerate(encoding.word_ids):
            word_id2subwords[word_id].append(self.tokenizer.decode(encoding.ids[token_id]))
        subwords_per_morpheme = [subwords for subwords in word_id2subwords.values()]
        assert len(subwords_per_morpheme) == len(
            morphemes
        ), f"inconsistent segmentation: {subwords_per_morpheme} / {morphemes}"

        readings: List[str] = []
        for morpheme, subwords in zip(morphemes, subwords_per_morpheme):
            readings.extend(self._align_morpheme(morpheme, subwords))
        return readings

    def _align_morpheme(self, morpheme: Morpheme, subwords: List[str]) -> List[str]:
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
        td_lattice: List[List[List[Node]]] = []
        td_holder: List[List[Tuple[Optional[Node], Optional[Node], float]]] = []
        node: Optional[Node] = None
        node_prev: Optional[Node]
        for i in range(len(surf)):
            td_lattice.append([])
            for j in range(len(reading) + 1):  # +1 for zero-width reading
                td_lattice[-1].append([])
        for i in range(len(surf) + 1):
            td_holder.append([])
            for j in range(len(reading) + 1):
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
    def _extend_kanji_reading_list(kanji_reading_list_orig: List[str]) -> List[str]:
        kanji_reading_list = []
        for kanji_reading in kanji_reading_list_orig:
            kanji_reading = re.sub("-", "", kanji_reading)
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


def get_word_level_readings(readings: List[str], tokens: List[str], subword_map: List[List[bool]]) -> List[str]:
    """サブワードレベルの読みを単語レベルの読みに変換．

    Args:
        readings: list[str] サブワードレベルの読み．
        tokens: list[str] サブワードレベルの表層．
        subword_map: list[list[bool]] subword_map[i][j] = True ならば i 番目の単語は j 番目のトークンを含む．
    """
    ret: List[str] = []
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
            ret.append("␣")
    return ret


def main():
    from argparse import ArgumentParser
    from collections import Counter
    from pathlib import Path

    from kwja.utils.constants import SPLIT_INTO_WORDS_MODEL_NAMES

    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name-or-path", type=str, help="model_name_or_path")
    parser.add_argument("-k", "--kanjidic", type=str, help="path to file")
    parser.add_argument("-i", "--input", type=str, help="path to input dir")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.model_name_or_path in SPLIT_INTO_WORDS_MODEL_NAMES:
        tokenizer_input_format: Literal["words", "text"] = "words"
    else:
        tokenizer_input_format = "text"
    kanjidic = KanjiDic(args.kanjidic)
    reading_aligner = ReadingAligner(tokenizer, tokenizer_input_format, kanjidic)

    reading_counter: Dict[str, int] = Counter()
    for path in Path(args.input).glob("**/*.knp"):
        logger.info(f"processing {path}")
        with path.open() as f:
            document = Document.from_knp(f.read())
        try:
            for reading in reading_aligner.align(document.morphemes):
                reading_counter[reading] += 1
        except ValueError:
            logger.warning(f"skip {document.doc_id} for an error")
    for subreading, count in sorted(
        sorted(reading_counter.items(), key=lambda pair: pair[0]), key=lambda pair: pair[1], reverse=True
    ):
        print(f"{subreading}\t{count}")


if __name__ == "__main__":
    main()
