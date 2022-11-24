import textwrap

from kwja.evaluators.word_module_metric import WordModuleMetric


def test_char_module_metric_perfect() -> None:
    metric = WordModuleMetric()
    predicted = textwrap.dedent(
        """\
        # S-ID:000-1
        * 1D
        + 1D
        風 _ 風 未定義語 15 その他 1 * 0 * 0
        が _ が 未定義語 15 その他 1 * 0 * 0
        * -1D
        + -1D
        吹く _ 吹く 未定義語 15 その他 1 * 0 * 0
        。 _ 。 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    gold = predicted
    metric.update([predicted], [gold])
    metric.update([predicted], [gold])  # Add the same data twice.
    # TODO: Run metric.compute() and check the result.
