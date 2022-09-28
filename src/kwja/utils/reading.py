import copy
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Final, Optional, Union

import jaconv
import numpy as np
from rhoknp import Document, Morpheme, Sentence
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.utils.kanjidic import KanjiDic

# KATAKANA-HIRAGANA PROLONGED SOUND MARK (0x30fc)
# "〜"(0x301C)  "⁓" (U+2053)、Full-width tilde:
# "～" (U+FF5E)、tilde operator: "∼" (U+223C)
# Half-widths HIRAGANA-KATAKANA PROLONGED SOUND MARK (U+FF70
CHOON_SET: Final = {"～", "〜", "∼", "⁓", "~", "ー", "ｰ", "-"}

HATSUON_SET: Final = {"っ", "ッ"}

LOWER2UPPER: Final = {
    "ぁ": "あ",
    "ぃ": "い",
    "ぅ": "う",
    "ぇ": "え",
    "ぉ": "お",
    "ゎ": "わ",
    "ヶ": "ケ",
    "ケ": "ヶ",
}

VOICED2VOICELESS: Final = {
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

PROLONGED_MAP: Final = {
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

PROLONGED_MAP_FOR_EROW: Final = {
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

IGNORE_READING = "IGNORED"
UNK = "[UNK]"
ID = "[ID]"
UNK_ID: Final = 0
ID_ID: Final = 1


def get_reading2id(path: str) -> dict[str, int]:
    reading2id = {UNK: UNK_ID, ID: ID_ID}
    with open(path, "r") as f:
        for line in f:
            if line := line.strip():
                if line not in reading2id:
                    reading2id[line] = len(reading2id)
    return reading2id


class ReadingAligner:
    DELIMITER = "▁"
    kana_re = re.compile("^[\u3041-\u30FF]+$")

    def __init__(self, tokenizer: PreTrainedTokenizerBase, kanji_dic: KanjiDic) -> None:
        self.tokenizer = tokenizer
        self.kanji_dic = kanji_dic

    def align(self, sequence: Union[Sentence, Document]) -> list[tuple[str, str]]:
        inp = " ".join([morpheme.surf for morpheme in sequence.morphemes])
        reading_list = []

        # assumption: morphemes are never combined
        subword_list = self.tokenizer.tokenize(inp)
        subwords_per_morpheme = []
        for subword in subword_list:
            if subword[0] == self.DELIMITER:
                subwords_per_morpheme.append([subword[1:]])
            else:
                assert len(subwords_per_morpheme) > 0
                subwords_per_morpheme[-1].append(subword)
        # assert(len(subwords_per_morpheme) == len(sequence.morphemes))
        if len(subwords_per_morpheme) != len(sequence.morphemes):
            logging.warning(f"something wrong with subword segmentation: {subword_list}")
            raise ValueError
        for morpheme, subwords in zip(sequence.morphemes, subwords_per_morpheme):
            reading_list.extend(self._align_morpheme(morpheme, subwords))
        assert len(subword_list) == len(reading_list)
        return list(zip(subword_list, reading_list))

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
            logging.warning(f"non-identical surf forms: {morpheme.surf}\t{surf}")

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
        td_holder: list[list[tuple[Optional[Node], Optional[Node], int]]] = []
        node: Optional[Node] = None
        node_prev: Optional[Node] = None
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
                if jaconv.kata2hira(ci) == cj:
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
                    reading_part = reading[j : j + len(kanji_reading)]  # noqa: E203
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
                        # TODO: combinatoin
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
                        for basenode in basenode_list:
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
            logging.warning("{}\t{}\t{}".format(seg, reading, subwords))

        # dummy
        posI, posJ = 0, 0
        subreading_list = []
        if empty_initial:
            subreading_list.append("")
        subreading = ""
        boundaries.pop(0)
        while True:
            wI, wJ = seg.pop(0)
            subreading += reading[posJ : posJ + wJ]  # noqa: E203
            posI += wI
            posJ += wJ
            if posI == boundaries[0]:
                subreading_list.append(subreading)
                if len(boundaries) > 1:
                    subreading = ""
                    boundaries.pop(0)
                else:
                    break
        return subreading_list

    def _extend_kanji_reading_list(self, kanji_reading_list_orig: list[str]) -> list[str]:
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
            ret.append("\u00A0")
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="model name or path to file")
    parser.add_argument("-k", "--kanjidic", type=str, help="path to file")
    parser.add_argument("-i", "--input", type=str, nargs="+", help="path glob (*.knp)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kanjidic = KanjiDic(args.kanjidic)
    aligner = ReadingAligner(tokenizer, kanjidic)

    import glob
    from collections import Counter

    subreading_counter: Dict[str, int] = Counter()
    for pathglob in args.input:
        for fpath in glob.glob(pathglob):
            logging.info(f"processing {fpath}")
            document = Document.from_knp(open(fpath).read())
            try:
                for subword, subreading in aligner.align(document):
                    # print(aligner.align(document))
                    subreading_counter[subreading] += 1
            except ValueError:
                logging.warn("skip {document.doc_id} for an error")
    for subreading, count in sorted(
        sorted(subreading_counter.items(), key=lambda pair: pair[0]), key=lambda pair: pair[1], reverse=True
    ):
        print(f"{subreading}\t{count}")
