import multiprocessing as mp
import re
import textwrap
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Dict, List, Optional, Tuple, Union

from kyoto_reader import KyotoReader
from rhoknp import KNP, Document, Jumanpp, Morpheme, Sentence
from rhoknp.props import FeatureDict, NamedEntity, NamedEntityCategory
from tqdm import tqdm

from kwja.utils.constants import BASE_PHRASE_FEATURES, IGNORE_VALUE_FEATURE_PAT, SUB_WORD_FEATURES

FEATURES_PAT = re.compile(r"(?P<features>(<[^>]+>)+)")
NE_OPTIONAL_PAT = re.compile(r"(?P<optional><NE:OPTIONAL:[^>]+>)")


class JumanppAugmenter:
    def __init__(self):
        self.jumanpp = Jumanpp(options=["--partial-input"])

    def augment_document(self, original_document: Document, update_original: bool = True) -> Document:
        buf = ""
        for sentence in original_document.sentences:
            buf += self._create_partial_input(sentence)

        with Popen(self.jumanpp.run_command, stdout=PIPE, stdin=PIPE, encoding="utf-8") as p:
            jumanpp_text, _ = p.communicate(input=buf)
        augmented_document = Document.from_jumanpp(jumanpp_text)

        for original_sentence, augmented_sentence in zip(original_document.sentences, augmented_document.sentences):
            self._postprocess_sentence(original_sentence, augmented_sentence, update_original=update_original)
        return augmented_document

    def augment_sentence(self, original_sentence: Sentence, update_original: bool = True) -> Sentence:
        buf = self._create_partial_input(original_sentence)
        with Popen(self.jumanpp.run_command, stdout=PIPE, stdin=PIPE, encoding="utf-8") as p:
            jumanpp_text, _ = p.communicate(input=buf)
        augmented_sentence = Sentence.from_jumanpp(jumanpp_text)

        self._postprocess_sentence(original_sentence, augmented_sentence, update_original=update_original)
        return augmented_sentence

    @staticmethod
    def _create_partial_input(sentence: Sentence) -> str:
        """
        create raw string for jumanpp --partial-input
        """
        buf = ""
        for morpheme in sentence.morphemes:
            buf += (
                f"\t{morpheme.surf}"
                f"\treading:{morpheme.reading}"
                f"\tbaseform:{morpheme.lemma}"
                f"\tpos:{morpheme.pos}"
                f"\tsubpos:{morpheme.subpos}"
                f"\tconjtype:{morpheme.conjtype}"
                f"\tconjform:{morpheme.conjform}\n"
            )
        buf += "\n"
        return buf

    @staticmethod
    def _postprocess_sentence(
        original_sentence: Sentence, augmented_sentence: Sentence, update_original: bool = True
    ) -> None:
        for original_morpheme, augmented_morpheme in zip(original_sentence.morphemes, augmented_sentence.morphemes):
            # Jumanpp may override reading
            assert augmented_morpheme.attributes is not None
            augmented_morpheme.attributes.reading = original_morpheme.reading
            if update_original and not original_sentence.need_knp:
                # add Semantics
                for k, v in augmented_morpheme.semantics.items():
                    original_morpheme.semantics[k] = v


def align(morphemes1: List[Morpheme], morphemes2: List[Morpheme]) -> Union[Dict[str, List[Morpheme]], None]:
    alignment = {}
    idx1, idx2 = 0, 0
    for _ in range(max(len(morphemes1), len(morphemes2))):
        if idx1 >= len(morphemes1) or idx2 >= len(morphemes2):
            break

        range1 = range(1, min(len(morphemes1) - idx1 + 1, 11))
        range2 = range(1, min(len(morphemes2) - idx2 + 1, 11))
        for i, j in product(range1, range2):
            subseq1, subseq2 = map(
                lambda x: "".join(morpheme.surf for morpheme in x),
                [morphemes1[idx1 : idx1 + i], morphemes2[idx2 : idx2 + j]],
            )
            if subseq1 == subseq2:
                key = "-".join(str(morpheme1.index) for morpheme1 in morphemes1[idx1 : idx1 + i])
                alignment[key] = morphemes2[idx2 : idx2 + j]
                idx1 += i
                idx2 += j
                break
        else:
            return None

    return alignment


def extract_named_entities(tagged_sentence: Sentence) -> List[Tuple[str, List[Morpheme]]]:
    named_entities = []
    category, morphemes_buff = "", []
    for morpheme in tagged_sentence.morphemes:
        if ne_tag := morpheme.semantics.get("NE"):
            assert isinstance(ne_tag, str)
            cat, span = ne_tag.split(":")
            if span in {"single", "head"}:
                category, morphemes_buff = cat, [morpheme]
            elif span in {"middle", "tail"} and cat == category:
                morphemes_buff.append(morpheme)

            if span in {"single", "tail"} and category != "":
                named_entities.append((category, morphemes_buff))
                category, morphemes_buff = "", []

    return named_entities


def set_named_entities(document: Document, sid2tagged_sentence: Dict[str, Sentence]) -> None:
    for sentence in document.sentences:
        # 既にneタグが付与されている文は対象としない
        if sentence.sid in sid2tagged_sentence and len(sentence.named_entities) == 0:
            tagged_sentence = sid2tagged_sentence[sentence.sid]
            alignment = align(tagged_sentence.morphemes, sentence.morphemes)
            if alignment is None:
                print(
                    f'alignment ({" ".join(m.surf for m in tagged_sentence.morphemes)} | '
                    f'{" ".join(m.surf for m in sentence.morphemes)}) not found'
                )
                continue

            for category, morphemes_buff in extract_named_entities(tagged_sentence):
                morphemes, keys = [], []
                for morpheme in morphemes_buff:
                    keys.append(str(morpheme.index))
                    if "-".join(keys) in alignment:
                        morphemes.extend(alignment["-".join(keys)])
                        keys = []

                if len(keys) == 0:
                    named_entity = NamedEntity(category=NamedEntityCategory(category), morphemes=morphemes)
                    sentence.named_entities.append(named_entity)
                else:
                    print(
                        f'morpheme span of {" ".join(m.surf for m in morphemes_buff)} not found in '
                        f'{" ".join(m.surf for m in sentence.morphemes)}'
                    )


def is_target_base_phrase_feature(k: str, v: Any) -> bool:
    name = k + (f":{v}" if isinstance(v, str) and IGNORE_VALUE_FEATURE_PAT.match(k) is None else "")
    return name in BASE_PHRASE_FEATURES


def refresh(document: Document) -> None:
    for morpheme in document.morphemes:
        feature_dict = FeatureDict()
        if morpheme.base_phrase.head == morpheme:
            feature_dict["基本句-主辞"] = True
        for feature in SUB_WORD_FEATURES:
            k, *vs = feature.split(":")
            if k in morpheme.features:
                feature_dict[k] = morpheme.features[k]
        morpheme.features = feature_dict
        morpheme.semantics.clear()

    for base_phrase in document.base_phrases:
        feature_dict = FeatureDict()
        for feature in BASE_PHRASE_FEATURES:
            k, *vs = feature.split(":")
            if k in base_phrase.features and is_target_base_phrase_feature(k, base_phrase.features[k]):
                feature_dict[k] = base_phrase.features[k]
        base_phrase.features = feature_dict

    for phrase in document.phrases:
        phrase.features.clear()

    for sentence in document.sentences:
        for ne1 in list(sentence.named_entities):
            span1 = {morpheme.index for morpheme in ne1.morphemes}
            for ne2 in sentence.named_entities:
                span2 = {morpheme.index for morpheme in ne2.morphemes}
                # あるnamed entityの一部もまたnamed entityである場合、外側だけ残す
                if len(span1 & span2) > 0 and len(span1) < len(span2):
                    print(
                        f'NE tag {" ".join(m.surf for m in ne1.morphemes)} removed '
                        f'due to the named entity {" ".join(m.surf for m in ne2.morphemes)} '
                        f"({sentence.sid}:{sentence.text})"
                    )
                    sentence.named_entities.remove(ne1)
                    break


def add_features(
    doc_ids: List[str],
    reader: KyotoReader,
    output_dir: Path,
    sid2tagged_sentence: Optional[Dict[str, Sentence]] = None,
) -> None:
    jumanpp_augmenter = JumanppAugmenter()
    knp = KNP(options=["-tab", "-dpnd-fast", "-read-feature"])

    for doc_id in tqdm(doc_ids):
        old_knp_lines = reader.get_knp(doc_id).split("\n")
        buf = []
        for idx, line in enumerate(old_knp_lines):
            if line.startswith("*") or line.startswith("+"):
                pass
            elif mo := FEATURES_PAT.search(line):
                features = mo.group("features")
                old_knp_lines[idx] = line.replace(features, "")
                buf.append((idx, features))

        document = Document.from_knp("\n".join(old_knp_lines))
        # 形態素意味情報付与 (引数に渡したdocumentをupdateする)
        _ = jumanpp_augmenter.augment_document(document)

        # 素性付与
        with Popen(knp.run_command, stdout=PIPE, stdin=PIPE, encoding="utf-8", errors="replace") as p:
            knp_text, _ = p.communicate(input=document.to_knp())

        new_knp_lines = knp_text.split("\n")
        # ダ列文語連体形など
        if len(new_knp_lines) != len(old_knp_lines):
            continue

        # 初めから付いていた素性の付与
        for idx, features in buf:
            assert new_knp_lines[idx].endswith(">")
            new_knp_lines[idx] += features
        knp_text = "\n".join(new_knp_lines)

        document = Document.from_knp(knp_text)
        if sid2tagged_sentence is not None:
            set_named_entities(document, sid2tagged_sentence)
        refresh(document)

        knp_text = document.to_knp()
        for mo in NE_OPTIONAL_PAT.finditer(knp_text):
            knp_text = knp_text.replace(mo.group("optional"), "")

        with output_dir.joinpath(f"{document.doc_id}.knp").open(mode="w") as f:
            f.write(knp_text)


def test_jumanpp_augmenter():
    jumanpp_augmenter = JumanppAugmenter()

    knp_text = textwrap.dedent(
        """\
        # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
        * 2D
        + 1D
        コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0
        + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        トス とす トス 名詞 6 サ変名詞 2 * 0 * 0
        を を を 助詞 9 格助詞 1 * 0 * 0
        * 2D
        + 3D
        ３ さん ３ 名詞 6 数詞 7 * 0 * 0
        回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0
        * -1D
        + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
        行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2
        。 。 。 特殊 1 句点 1 * 0 * 0
        EOS
        """
    )
    sentence = Sentence.from_knp(knp_text)
    _ = jumanpp_augmenter.augment_sentence(sentence)
    expected = textwrap.dedent(
        """\
        # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
        * 2D
        + 1D
        コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0 "自動獲得:Wikipedia Wikipediaリダイレクト:硬貨"
        + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        トス とす トス 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:トス/とす ドメイン:スポーツ カテゴリ:抽象物"
        を を を 助詞 9 格助詞 1 * 0 * 0 "代表表記:を/を"
        * 2D
        + 3D
        ３ さん ３ 名詞 6 数詞 7 * 0 * 0 "代表表記:３/さん カテゴリ:数量"
        回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 "代表表記:回/かい 準内容語"
        * -1D
        + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
        行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2 "代表表記:行う/おこなう"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        """
    )
    assert sentence.to_knp() == expected

    knp_text = textwrap.dedent(
        """\
        # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
        * 2D
        + 1D
        コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0
        + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        トス とす トス 名詞 6 サ変名詞 2 * 0 * 0
        を を を 助詞 9 格助詞 1 * 0 * 0
        * 2D
        + 3D
        ３ さん ３ 名詞 6 数詞 7 * 0 * 0
        回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0
        * -1D
        + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
        行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2
        。 。 。 特殊 1 句点 1 * 0 * 0
        EOS
        # S-ID:w201106-0000060050-2 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-64.95916 MOD:2013/04/13
        * 1D
        + 1D <rel type="ノ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        表 おもて 表 名詞 6 普通名詞 1 * 0 * 0
        が が が 助詞 9 格助詞 1 * 0 * 0
        * 2D
        + 2D <rel type="ガ" target="表" sid="w201106-0000060050-2" id="0"/><rel type="外の関係" target="数" sid="w201106-0000060050-2" id="2"/>
        出た でた 出る 動詞 2 * 0 母音動詞 1 タ形 10
        * 5D
        + 5D <rel type="ノ" target="出た" sid="w201106-0000060050-2" id="1"/>
        数 かず 数 名詞 6 普通名詞 1 * 0 * 0
        だけ だけ だけ 助詞 9 副助詞 2 * 0 * 0
        、 、 、 特殊 1 読点 2 * 0 * 0
        * 4D
        + 4D
        フィールド ふぃーるど フィールド 名詞 6 普通名詞 1 * 0 * 0
        上 じょう 上 接尾辞 14 名詞性名詞接尾辞 2 * 0 * 0
        の の の 助詞 9 接続助詞 3 * 0 * 0
        * 5D
        + 5D <rel type="修飾" target="フィールド上" sid="w201106-0000060050-2" id="3"/><rel type="修飾" mode="AND" target="数" sid="w201106-0000060050-2" id="2"/>
        モンスター もんすたー モンスター 名詞 6 普通名詞 1 * 0 * 0
        を を を 助詞 9 格助詞 1 * 0 * 0
        * -1D
        + -1D <rel type="ヲ" target="モンスター" sid="w201106-0000060050-2" id="4"/><rel type="ガ" target="不特定:状況"/>
        破壊 はかい 破壊 名詞 6 サ変名詞 2 * 0 * 0
        する する する 動詞 2 * 0 サ変動詞 16 基本形 2
        。 。 。 特殊 1 句点 1 * 0 * 0
        EOS
        """
    )
    document = Document.from_knp(knp_text)
    _ = jumanpp_augmenter.augment_document(document)
    expected = textwrap.dedent(
        """\
        # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
        * 2D
        + 1D
        コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0 "自動獲得:Wikipedia Wikipediaリダイレクト:硬貨"
        + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        トス とす トス 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:トス/とす ドメイン:スポーツ カテゴリ:抽象物"
        を を を 助詞 9 格助詞 1 * 0 * 0 "代表表記:を/を"
        * 2D
        + 3D
        ３ さん ３ 名詞 6 数詞 7 * 0 * 0 "代表表記:３/さん カテゴリ:数量"
        回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 "代表表記:回/かい 準内容語"
        * -1D
        + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
        行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2 "代表表記:行う/おこなう"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        # S-ID:w201106-0000060050-2 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-64.95916 MOD:2013/04/13
        * 1D
        + 1D <rel type="ノ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        表 おもて 表 名詞 6 普通名詞 1 * 0 * 0 "代表表記:表/おもて カテゴリ:場所-機能 漢字読み:訓"
        が が が 助詞 9 格助詞 1 * 0 * 0 "代表表記:が/が"
        * 2D
        + 2D <rel type="ガ" target="表" sid="w201106-0000060050-2" id="0"/><rel type="外の関係" target="数" sid="w201106-0000060050-2" id="2"/>
        出た でた 出る 動詞 2 * 0 母音動詞 1 タ形 10 "代表表記:出る/でる 反義:動詞:入る/はいる 自他動詞:他:出す/だす 補文ト"
        * 5D
        + 5D <rel type="ノ" target="出た" sid="w201106-0000060050-2" id="1"/>
        数 かず 数 名詞 6 普通名詞 1 * 0 * 0 "代表表記:数/かず カテゴリ:数量 漢字読み:訓"
        だけ だけ だけ 助詞 9 副助詞 2 * 0 * 0 "代表表記:だけ/だけ"
        、 、 、 特殊 1 読点 2 * 0 * 0 "代表表記:、/、"
        * 4D
        + 4D
        フィールド ふぃーるど フィールド 名詞 6 普通名詞 1 * 0 * 0 "代表表記:フィールド/ふぃーるど カテゴリ:場所-その他"
        上 じょう 上 接尾辞 14 名詞性名詞接尾辞 2 * 0 * 0 "代表表記:上/じょう"
        の の の 助詞 9 接続助詞 3 * 0 * 0 "代表表記:の/の"
        * 5D
        + 5D <rel type="修飾" target="フィールド上" sid="w201106-0000060050-2" id="3"/><rel type="修飾" mode="AND" target="数" sid="w201106-0000060050-2" id="2"/>
        モンスター もんすたー モンスター 名詞 6 普通名詞 1 * 0 * 0 "代表表記:モンスター/もんすたー カテゴリ:人"
        を を を 助詞 9 格助詞 1 * 0 * 0 "代表表記:を/を"
        * -1D
        + -1D <rel type="ヲ" target="モンスター" sid="w201106-0000060050-2" id="4"/><rel type="ガ" target="不特定:状況"/>
        破壊 はかい 破壊 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:破壊/はかい カテゴリ:抽象物 反義:名詞-サ変名詞:建設/けんせつ"
        する する する 動詞 2 * 0 サ変動詞 16 基本形 2 "代表表記:する/する 自他動詞:自:成る/なる 付属動詞候補（基本）"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        """
    )
    assert document.to_knp() == expected


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT", type=str, help="path to input knp dir")
    parser.add_argument("OUTPUT", type=str, help="path to output dir")
    parser.add_argument("--ne-tags", default=None, type=str, help="path to ne tags")
    parser.add_argument("-j", default=1, type=int, help="number of jobs")
    args = parser.parse_args()

    reader = KyotoReader(args.INPUT, did_from_sid=True)
    output_dir = Path(args.OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ne_tags:
        with open(args.ne_tags, mode="r") as f:
            document = Document.from_jumanpp(f.read())
        sid2tagged_sentence = {sentence.sid: sentence for sentence in document.sentences}
    else:
        sid2tagged_sentence = None

    doc_ids = reader.doc_ids
    chunk_size = len(doc_ids) // args.j + int(len(doc_ids) % args.j > 0)
    iterable = [
        (doc_ids[slice(start, start + chunk_size)], reader, output_dir, sid2tagged_sentence)
        for start in range(0, len(doc_ids), chunk_size)
    ]
    with mp.Pool(args.j) as pool:
        pool.starmap(add_features, iterable)


if __name__ == "__main__":
    main()
