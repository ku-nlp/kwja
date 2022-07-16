import itertools

from rhoknp.units.utils import DepType

TYPO_OPN2TOKEN = {
    "K": "<k>",
    "D": "<d>",
    "_": "<_>",
}
TOKEN2TYPO_OPN = {v: k for k, v in TYPO_OPN2TOKEN.items()}
TYPO_DUMMY_TOKEN = "<dummy>"
ENE_TYPES = (
    "0",  # CONCEPT
    "1.0",  # 名前＿その他
    "1.1",  # 人名
    "1.2",  # 神名
    "1.3",  # 動物呼称名
    "1.4",  # 組織名
    "1.5",  # 地名
    "1.6",  # 施設名
    "1.7",  # 製品名
    "1.9",  # イベント名
    "1.10",  # 自然物名
    "1.11",  # 病気名
    "1.12",  # 色名
)
ENE_TYPE_BIES: tuple = ("PAD",) + tuple(
    itertools.chain.from_iterable((f"B-{ene}", f"I-{ene}", f"E-{ene}") for ene in ENE_TYPES)
)

SEG_TYPES = (
    "B",
    "I",
)
INDEX2SEG_TYPE = {index: seg_type for index, seg_type in enumerate(SEG_TYPES)}

# 品詞
POS_TYPES = (
    "助動詞",
    "接続詞",
    "連体詞",
    "形容詞",
    "動詞",
    "特殊",
    "助詞",
    "判定詞",
    "接尾辞",
    "副詞",
    "名詞",
    "指示詞",
    "感動詞",
    "接頭辞",
)
INDEX2POS_TYPE = {index: pos_type for index, pos_type in enumerate(POS_TYPES)}
# 品詞細分類
SUBPOS_TYPES = (
    "格助詞",
    "その他",
    "形式名詞",
    "形容詞性述語接尾辞",
    "名詞性名詞接尾辞",
    "未対応表現",
    "普通名詞",
    "イ形容詞接頭辞",
    "人名",
    "組織名",
    "形容詞性名詞接尾辞",
    "名詞接頭辞",
    "固有名詞",
    "括弧終",
    "動詞性接尾辞",
    "記号",
    "*",
    "接続助詞",
    "終助詞",
    "名詞形態指示詞",
    "地名",
    "数詞",
    "名詞性名詞助数辞",
    "副助詞",
    "ナ形容詞接頭辞",
    "副詞的名詞",
    "句点",
    "サ変名詞",
    "副詞形態指示詞",
    "括弧始",
    "連体詞形態指示詞",
    "名詞性述語接尾辞",
    "動詞接頭辞",
    "読点",
    "名詞性特殊接尾辞",
    "時相名詞",
)
INDEX2SUBPOS_TYPE = {index: subpos_type for index, subpos_type in enumerate(SUBPOS_TYPES)}
# 活用型
CONJTYPE_TYPES = (
    "助動詞だろう型",
    "子音動詞ワ行",
    "カ変動詞",
    "子音動詞カ行促音便形",
    "ナ形容詞特殊",
    "子音動詞マ行",
    "なり列",
    "子音動詞ワ行文語音便形",
    "動詞性接尾辞うる型",
    "イ形容詞アウオ段",
    "助動詞そうだ型",
    "子音動詞カ行",
    "子音動詞ラ行",
    "*",
    "助動詞く型",
    "サ変動詞",
    "子音動詞ラ行イ形",
    "カ変動詞来",
    "ナ形容詞",
    "方言",
    "イ形容詞イ段",
    "子音動詞タ行",
    "子音動詞サ行",
    "判定詞",
    "子音動詞ガ行",
    "母音動詞",
    "助動詞ぬ型",
    "子音動詞ナ行",
    "無活用型",
    "ザ変動詞",
    "ナノ形容詞",
    "動詞性接尾辞ます型",
    "タル形容詞",
    "イ形容詞イ段特殊",
    "子音動詞バ行",
)
INDEX2CONJTYPE_TYPE = {index: conjtype_type for index, conjtype_type in enumerate(CONJTYPE_TYPES)}
# 活用形
CONJFORM_TYPES = (
    "ダ列基本条件形",
    "デス列タ系条件形",
    "ダ列タ系条件形",
    "古語基本形(なり)",
    "デアル列基本条件形",
    "デス列基本推量形",
    "デス列タ系連用テ形",
    "命令形",
    "タ系連用チャ形",
    "デアル列タ系連用タリ形",
    "基本連用形",
    "タ系連用テ形",
    "文語已然形",
    "タ接連用形",
    "デアル列連用形",
    "デアル列基本形",
    "ダ列タ系推量形",
    "ダ列基本連体形",
    "デアル列基本連用形",
    "文語基本形",
    "ヤ列基本推量形",
    "省略意志形",
    "デアル列命令形",
    "文語連体形",
    "デス列タ形",
    "タ形",
    "音便条件形",
    "デアル列タ形",
    "ダ列タ系連用タリ形",
    "未然形",
    "ダ列タ形",
    "文語命令形",
    "語幹",
    "文語連用形",
    "語幹異形",
    "タ系条件形",
    "音便基本形",
    "文語未然形",
    "ダ列文語基本形",
    "ダ列基本推量形",
    "デアル列タ系条件形",
    "ダ列基本連用形",
    "ダ列特殊連体形",
    "基本形",
    "ダ列基本省略推量形",
    "ヤ列基本形",
    "デアル列タ系推量形",
    "ダ列タ系連用ジャ形",
    "ダ列文語連体形",
    "*",
    "デアル列タ系連用テ形",
    "ダ列タ系連用テ形",
    "タ系連用チャ形２",
    "音便条件形２",
    "基本推量形",
    "タ系推量形",
    "デアル列基本推量形",
    "デス列基本形",
    "基本条件形",
    "意志形",
    "ダ列特殊連用形",
    "タ系連用タリ形",
    "エ基本形",
    "デス列基本省略推量形",
)
INDEX2CONJFORM_TYPE = {index: conjform_type for index, conjform_type in enumerate(CONJFORM_TYPES)}

WORD_FEATURES = (
    "基本句-主辞",
    "基本句-区切",
    "文節-区切",
)


BASE_PHRASE_FEATURES = (
    # type
    "用言:動",
    "用言:形",
    "用言:判",
    "体言",
    # modality
    "モダリティ-疑問",
    "モダリティ-意志",
    "モダリティ-勧誘",
    "モダリティ-命令",
    "モダリティ-禁止",
    "モダリティ-評価:弱",
    "モダリティ-評価:強",
    "モダリティ-認識-推量",
    "モダリティ-認識-蓋然性",
    "モダリティ-認識-証拠性",
    "モダリティ-依頼Ａ",
    "モダリティ-依頼Ｂ",
    # tense
    "時制:過去",
    "時制:非過去",
    # negation
    "否定表現",
    # clause
    "節-主辞",
    "節-区切",
)
IGNORE_INDEX = -100


DEPENDENCY_TYPE2INDEX: dict[DepType, int] = {
    DepType.DEPENDENCY: 0,
    DepType.PARALLEL: 1,
    DepType.APPOSITION: 2,
    DepType.IMPERFECT_PARALLEL: 3,
}
INDEX2DEPENDENCY_TYPE: dict[int, DepType] = {
    index: dependency_type for dependency_type, index in DEPENDENCY_TYPE2INDEX.items()
}


DISCOURSE_RELATIONS = (
    "談話関係なし",
    "原因・理由",
    "目的",
    "条件",
    "根拠",
    "対比",
    "逆接",
)
