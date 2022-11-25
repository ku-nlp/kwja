import textwrap

from kwja.evaluators.char_module_metric import CharModuleMetric


def test_update() -> None:
    metric = CharModuleMetric()
    gold = textwrap.dedent(
        """\
        # S-ID:000-1
        風 _ 風 未定義語 15 その他 1 * 0 * 0
        が _ が 未定義語 15 その他 1 * 0 * 0
        吹く _ 吹く 未定義語 15 その他 1 * 0 * 0
        。 _ 。 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    predicted = gold
    gold = predicted
    metric.update([predicted], [gold])
    metric.update([predicted], [gold])


def test_evaluate_word_segmentation_0() -> None:
    metric = CharModuleMetric()
    gold = textwrap.dedent(
        """\
        # S-ID:000-1
        風 _ 風 未定義語 15 その他 1 * 0 * 0
        が _ が 未定義語 15 その他 1 * 0 * 0
        吹く _ 吹く 未定義語 15 その他 1 * 0 * 0
        。 _ 。 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    predicted = gold
    actual = metric.evaluate_word_segmentation(predicted, gold)
    expected = {
        "word_segmentation/acc": 1.0,
        "word_segmentation/f1": 1.0,
    }
    assert actual == expected


def test_evaluate_word_segmentation_1() -> None:
    metric = CharModuleMetric()
    predicted = textwrap.dedent(
        """\
        # S-ID:000-1
        風 _ 風 未定義語 15 その他 1 * 0 * 0
        が _ が 未定義語 15 その他 1 * 0 * 0
        吹 _ 吹 未定義語 15 その他 1 * 0 * 0
        く _ く 未定義語 15 その他 1 * 0 * 0
        。 _ 。 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    gold = textwrap.dedent(
        """\
        # S-ID:000-1
        風 _ 風 未定義語 15 その他 1 * 0 * 0
        が _ が 未定義語 15 その他 1 * 0 * 0
        吹く _ 吹く 未定義語 15 その他 1 * 0 * 0
        。 _ 。 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    actual = metric.evaluate_word_segmentation(predicted, gold)
    expected = {
        "word_segmentation/acc": 4 / 5,
        "word_segmentation/f1": (2 * (3 / 5) * (3 / 4)) / (3 / 5 + 3 / 4),  # tp: 3, fp: 1, fn: 1
    }
    assert actual == expected
