from kwja.evaluators.typo_module_metric import TypoModuleMetric


def test_char_module_metric_perfect() -> None:
    metric = TypoModuleMetric()
    predicted = "風が吹く。"
    gold = predicted
    metric.update([predicted], [gold])
    metric.update([predicted], [gold])  # Add the same data twice.
    # TODO: Run metric.compute() and check the result.
