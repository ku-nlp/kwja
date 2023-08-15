import multiprocessing as mp
import re
import subprocess
import textwrap
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Dict, List, Optional, Tuple

from rhoknp import KNP, Document, Jumanpp, Morpheme, Sentence
from rhoknp.props import FeatureDict, NamedEntity, NamedEntityCategory
from rhoknp.utils.reader import chunk_by_document, chunk_by_sentence

from kwja.utils.constants import BASE_PHRASE_FEATURES, IGNORE_VALUE_FEATURE_PAT, SUB_WORD_FEATURES
from kwja.utils.logging_util import track


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
        alignment = align_morphemes(original_sentence.morphemes, augmented_sentence.morphemes)
        if alignment is None:
            return None
        keys = []
        for original_morpheme in original_sentence.morphemes:
            keys.append(str(original_morpheme.index))
            if "-".join(keys) in alignment:
                aligned = alignment["-".join(keys)]
                if len(keys) == 1 and len(aligned) == 1:
                    augmented_morpheme = aligned[0]
                    # Jumanpp may override reading
                    augmented_morpheme.reading = original_morpheme.reading
                    if update_original and not original_sentence.need_knp:
                        original_morpheme.semantics.update(augmented_morpheme.semantics)
                keys = []


def align_morphemes(morphemes1: List[Morpheme], morphemes2: List[Morpheme]) -> Optional[Dict[str, List[Morpheme]]]:
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
            alignment = align_morphemes(tagged_sentence.morphemes, sentence.morphemes)
            if alignment is None:
                print(
                    f'alignment ({" ".join(morpheme.surf for morpheme in tagged_sentence.morphemes)} | '
                    f'{" ".join(morpheme.surf for morpheme in sentence.morphemes)}) not found'
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
                    morphemes[-1].base_phrase.features["NE"] = f"{named_entity.category.value}:{named_entity.text}"
                else:
                    print(
                        f'morpheme span of {" ".join(morpheme.surf for morpheme in morphemes_buff)} not found in '
                        f'{" ".join(morpheme.surf for morpheme in sentence.morphemes)}'
                    )


def is_target_base_phrase_feature(k: str, v: Any) -> bool:
    name = k + (f":{v}" if isinstance(v, str) and IGNORE_VALUE_FEATURE_PAT.match(k) is None else "")
    return name in BASE_PHRASE_FEATURES


def refresh(document: Document) -> None:
    keys = [feature.split(":")[0] for feature in SUB_WORD_FEATURES]
    for morpheme in document.morphemes:
        feature_dict = FeatureDict()
        if morpheme.base_phrase.head == morpheme:
            feature_dict["基本句-主辞"] = True
        feature_dict.update({key: morpheme.features[key] for key in keys if key in morpheme.features})
        morpheme.features = feature_dict

    keys = [feature.split(":")[0] for feature in BASE_PHRASE_FEATURES]
    for base_phrase in document.base_phrases:
        feature_dict = FeatureDict()
        if (
            (feature := base_phrase.features.get("NE"))
            and isinstance(feature, str)
            and feature.startswith("OPTIONAL") is False
        ):
            feature_dict["NE"] = feature
        feature_dict.update(
            {
                key: base_phrase.features[key]
                for key in keys
                if key in base_phrase.features and is_target_base_phrase_feature(key, base_phrase.features[key])
            }
        )
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
                        f'NE tag {" ".join(morpheme.surf for morpheme in ne1.morphemes)} removed '
                        f'due to the named entity {" ".join(morpheme.surf for morpheme in ne2.morphemes)} '
                        f"({sentence.sid}:{sentence.text})"
                    )
                    del ne1.morphemes[-1].base_phrase.features["NE"]
                    break


def assign_features_and_save(
    knp_texts: List[str],
    output_root: Path,
    doc_id2split: Dict[str, str],
    sid2tagged_sentence: Optional[Dict[str, Sentence]] = None,
) -> None:
    jumanpp_augmenter = JumanppAugmenter()
    knp = KNP(options=["-tab", "-dpnd-fast", "-read-feature"])
    for knp_text in track(knp_texts):
        try:
            document = Document.from_knp(knp_text)
        except ValueError:
            print("ignore broken knp file")
            continue
        if document.doc_id not in doc_id2split:
            continue

        buf = []
        for morpheme in document.morphemes:
            buf.append({**morpheme.features})
            morpheme.features.clear()

        # 形態素意味情報付与 (引数に渡したdocumentをupdateする)
        _ = jumanpp_augmenter.augment_document(document)

        # 素性付与
        with Popen(knp.run_command, stdout=PIPE, stdin=PIPE, encoding="utf-8", errors="replace") as p:
            assigned_knp_text, _ = p.communicate(input=document.to_knp())
        # ダ列文語連体形など
        if len(assigned_knp_text.split("\n")) != len(knp_text.split("\n")):
            continue
        document = Document.from_knp(assigned_knp_text)

        # 初めから付いていた素性の付与
        for morpheme, features in zip(document.morphemes, buf):
            morpheme.features.update(features)

        if sid2tagged_sentence is not None:
            set_named_entities(document, sid2tagged_sentence)

        refresh(document)

        doc_id = document.doc_id
        split = doc_id2split[doc_id]
        output_root.joinpath(f"{split}/{doc_id}.knp").write_text(document.to_knp())


def test_jumanpp_version():
    out = subprocess.run(["jumanpp", "--version"], capture_output=True, encoding="utf-8", text=True)
    match = re.match(r"Juman\+\+ Version: 2\.0\.0-dev\.(\d{8}).+", out.stdout)
    assert match is not None and int(match.group(1)) >= 20220605, "Juman++ version is old. Please update Juman++."


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
    parser.add_argument("--id", type=str, help="path to id")
    parser.add_argument("--ne-tags", default=None, type=str, help="path to ne tags")
    parser.add_argument("-j", default=1, type=int, help="number of jobs")
    parser.add_argument(
        "--doc-id-format", default="default", type=str, help="doc id format to identify document boundary"
    )
    args = parser.parse_args()

    knp_texts = []
    for input_file in Path(args.INPUT).glob("**/*.knp"):
        with input_file.open(mode="r") as f:
            knp_texts += [knp_text for knp_text in chunk_by_document(f, doc_id_format=args.doc_id_format)]

    if args.ne_tags:
        with open(args.ne_tags, mode="r") as f:
            sentences = [Sentence.from_jumanpp(jumanpp_text) for jumanpp_text in chunk_by_sentence(f)]
        sid2tagged_sentence = {sentence.sid: sentence for sentence in sentences}
    else:
        sid2tagged_sentence = None

    output_root = Path(args.OUTPUT)
    doc_id2split = {}
    for id_file in Path(args.id).glob("*.id"):
        if output_root.parts[-1] == "kyoto_ed":
            if id_file.stem != "all":
                continue
        else:
            if id_file.stem not in {"train", "dev", "test"}:
                continue
        split = "valid" if id_file.stem == "dev" else id_file.stem
        output_root.joinpath(split).mkdir(parents=True, exist_ok=True)
        for doc_id in id_file.read_text().splitlines():
            doc_id2split[doc_id] = split

    chunk_size = len(knp_texts) // args.j + int(len(knp_texts) % args.j > 0)
    iterable = [
        (knp_texts[slice(start, start + chunk_size)], output_root, doc_id2split, sid2tagged_sentence)
        for start in range(0, len(knp_texts), chunk_size)
    ]
    with mp.Pool(args.j) as pool:
        pool.starmap(assign_features_and_save, iterable)


if __name__ == "__main__":
    main()
