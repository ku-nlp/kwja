from kwja.utils.reading_prediction import get_word_level_readings


def test_get_word_level_readings():
    readings = ["[UNK]", "[ID]", "[ID]", "べんきょう"]
    tokens = ["[CLS]", "テス", "ト", "勉強"]
    subword_map = [
        [False, True, True, False],
        [False, False, False, True],
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert get_word_level_readings(readings, tokens, subword_map) == ["テスト", "べんきょう"]
