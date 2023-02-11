# import textwrap

# from kwja.evaluators.word_module_metric import WordModuleMetric

# TODO
# def test_word_module_metric() -> None:
#     metric = WordModuleMetric()
#     predicted = textwrap.dedent(
#         """\
#         # S-ID:000-1
#         * 1D
#         + 1D
#         風 _ 風 未定義語 15 その他 1 * 0 * 0
#         が _ が 未定義語 15 その他 1 * 0 * 0
#         * -1D
#         + -1D
#         吹く _ 吹く 未定義語 15 その他 1 * 0 * 0
#         。 _ 。 未定義語 15 その他 1 * 0 * 0
#         EOS
#         """
#     )
#     gold = predicted
#     metric.update([predicted], [gold], [0])
#     metric.update([predicted], [gold], [1])  # Add the same data twice.
