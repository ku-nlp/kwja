import pytest
from rhoknp import Morpheme

from jula.utils.word_normalize import MorphemeDenormalizer, MorphemeNormalizer, get_normalization_opns, get_normalized

wellformed_list = [
    ("なぁ", ["K", "S"], "なあ"),
    ("なー", ["K", "D"], "な"),
    ("な〜", ["K", "D"], "な"),
    ("おっ", ["K", "D"], "お"),
    ("さぁ", ["K", "S"], "さあ"),
    ("だぁ", ["K", "D"], "だ"),
    ("ねぇ", ["K", "S"], "ねえ"),
    ("うむっ", ["K", "K", "D"], "うむ"),
    ("のッ", ["K", "D"], "の"),
    ("バイキングっ", ["K", "K", "K", "K", "K", "D"], "バイキング"),
    ("楽しむっ", ["K", "K", "K", "D"], "楽しむ"),
    ("あっっっっっ", ["K", "K", "D", "D", "D", "D"], "あっ"),
    ("ますぅ", ["K", "K", "D"], "ます"),
    ("かい〜", ["K", "K", "D"], "かい"),
    ("そう〜", ["K", "K", "D"], "そう"),
    ("おーい", ["K", "D", "K"], "おい"),
    ("あーーー", ["K", "K", "D", "D"], "あー"),
    ("自然ー", ["K", "K", "D"], "自然"),
    ("きたー", ["K", "K", "D"], "きた"),
    ("ましたー", ["K", "K", "K", "D"], "ました"),
    ("ました〜", ["K", "K", "K", "D"], "ました"),
    ("飲んだー", ["K", "K", "K", "D"], "飲んだ"),
    ("続くー", ["K", "K", "D"], "続く"),
    ("寒〜い", ["K", "D", "K"], "寒い"),
    ("いや〜", ["K", "K", "D"], "いや"),
    ("あっちこち", ["K", "D", "K", "K", "K"], "あちこち"),
    ("かるーい", ["K", "K", "D", "K"], "かるい"),
    ("ずーっと", ["K", "P", "K", "K"], "ずうっと"),
    ("よー", ["K", "P"], "よう"),
    ("もーれつ", ["K", "P", "K", "K"], "もうれつ"),
    ("咲いたー", ["K", "K", "K", "D"], "咲いた"),
    ("なぁ〜", ["K", "S", "D"], "なあ"),
    ("ふわーっと", ["K", "K", "D", "K", "K"], "ふわっと"),
    ("ずら〜っと", ["K", "K", "D", "K", "K"], "ずらっと"),
    ("でーす", ["K", "D", "K"], "です"),
    ("で〜す", ["K", "D", "K"], "です"),
    ("まーす", ["K", "D", "K"], "ます"),
    ("ま〜す", ["K", "D", "K"], "ます"),
    ("いや〜", ["K", "K", "D"], "いや"),
    ("やったぁー", ["K", "K", "K", "D", "D"], "やった"),
    ("安っぽぃ", ["K", "K", "K", "S"], "安っぽい"),
    ("安っぽぃー", ["K", "K", "K", "S", "D"], "安っぽい"),
    ("だー", ["K", "D"], "だ"),
    ("ね〜", ["K", "E"], "ねえ"),
    ("すっげー", ["K", "D", "K", "E"], "すげえ"),
    ("まぁまぁだ", ["K", "S", "K", "S", "K"], "まあまあだ"),
    ("びみょーーー", ["K", "K", "K", "P", "D", "D"], "びみょう"),
    ("ごめんなさーぃ", ["K", "K", "K", "K", "K", "D", "S"], "ごめんなさい"),
    ("あっまい", ["K", "D", "K", "K"], "あまい"),
    ("もどかしー", ["K", "K", "K", "K", "P"], "もどかしい"),
    ("かわぃぃー", ["K", "K", "S", "S", "D"], "かわいい"),
    ("ほどよぃー", ["K", "K", "K", "S", "D"], "ほどよい"),
    ("あぉーい", ["K", "S", "D", "K"], "あおい"),
    ("鎌ヶ谷", ["K", "S", "K"], "鎌ケ谷"),
    ("龍ヶ崎", ["K", "S", "K"], "龍ケ崎"),
    ("八ヶ岳", ["K", "S", "K"], "八ケ岳"),
    ("湯ヶ島", ["K", "S", "K"], "湯ケ島"),
    ("がえる", ["V", "K", "K"], "かえる"),
    ("がえるー", ["V", "K", "K", "D"], "かえる"),
]


@pytest.mark.parametrize("surf,ops,expected", wellformed_list)
def test_gen_ormalized_surf(surf, ops, expected):
    assert get_normalized(surf, ops, strict=True) == expected


@pytest.mark.parametrize("surf,expected,normalized", wellformed_list)
def test_get_normalization_opns(surf, expected, normalized):
    opns = get_normalization_opns(surf, normalized)
    assert len(opns) == len(expected)
    assert all([a == b for a, b in zip(opns, expected)])


malformed_list = [
    ("なあ", ["K", "S"], "なあ"),
    ("なー", ["K", "S"], "なー"),
    ("ー", ["P"], "ー"),
]


@pytest.mark.parametrize("surf,ops,expected", malformed_list)
def test_gen_ormalized_surf_malformed(surf, ops, expected):
    with pytest.raises(ValueError):
        get_normalized(surf, ops, strict=True)


@pytest.mark.parametrize("surf,ops,expected", malformed_list)
def test_gen_ormalized_surf_malformed_loose(surf, ops, expected):
    assert get_normalized(surf, ops, strict=False) == expected


def test_morpheme_normalizer():
    jumanpp_text = 'きたー きた きる 動詞 2 * 0 母音動詞 1 タ形 10 "代表表記:着る/きる ドメイン:家庭・暮らし 反義:動詞:脱ぐ/ぬぐ 非標準表記:DPL"'
    expected = ["K", "K", "D"]
    morpheme = Morpheme.from_jumanpp(jumanpp_text)
    normalizer = MorphemeNormalizer()
    opns = normalizer.get_normalization_opns(morpheme)
    assert len(opns) == len(expected)
    assert all([a == b for a, b in zip(opns, expected)])


denormalize_list = [
    ("なあ", "なぁ"),
    ("さあ", "さぁ"),
    ("もうれつ", "もーれつ"),
    ("鎌ケ谷", "鎌ヶ谷"),
    ("八ケ岳", "八ヶ岳"),
]


@pytest.mark.parametrize("surf,expected", denormalize_list)
def test_denofrmalize_deterministic(surf, expected):
    md = MorphemeDenormalizer()
    assert md._denormalize_deterministic(surf) == expected
