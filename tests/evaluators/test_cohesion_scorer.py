import io
import json
from pathlib import Path

from kwja.metrics.cohesion_scorer import Metric, Scorer


def test_scorer(fixture_data_dir: Path, fixture_scorer: Scorer):
    expected_scores = json.loads(fixture_data_dir.joinpath("expected/cohesion_score.json").read_text())
    score_dict = fixture_scorer.run().to_dict()
    for case in fixture_scorer.pas_cases:
        case_result = score_dict[case]
        for analysis in Scorer.ARGUMENT_TYPE2ANALYSIS.values():
            expected: dict = expected_scores[case][analysis]
            actual: Metric = case_result[analysis]
            assert expected["tp_fp"] == actual.tp_fp
            assert expected["tp_fn"] == actual.tp_fn
            assert expected["tp"] == actual.tp


def test_score_result_add(fixture_data_dir: Path, fixture_scorer: Scorer):
    expected_scores = json.loads(fixture_data_dir.joinpath("expected/cohesion_score.json").read_text())
    score_result1 = fixture_scorer.run()
    score_result2 = fixture_scorer.run()
    score_result = score_result1 + score_result2
    score_dict = score_result.to_dict()
    for case in fixture_scorer.pas_cases:
        case_result = score_dict[case]
        for analysis in Scorer.ARGUMENT_TYPE2ANALYSIS.values():
            expected: dict = expected_scores[case][analysis]
            actual: Metric = case_result[analysis]
            assert actual.tp_fp == expected["tp_fp"] * 2
            assert actual.tp_fn == expected["tp_fn"] * 2
            assert actual.tp == expected["tp"] * 2


def test_export_txt(fixture_data_dir: Path, fixture_scorer: Scorer):
    score_result = fixture_scorer.run()
    with io.StringIO() as string:
        score_result.export_txt(string)
        string_actual = string.getvalue()
    string_expected = fixture_data_dir.joinpath("expected/cohesion_score.txt").read_text()
    assert string_actual == string_expected


def test_export_csv(fixture_data_dir: Path, fixture_scorer: Scorer):
    score_result = fixture_scorer.run()
    with io.StringIO() as string:
        score_result.export_csv(string)
        string_actual = string.getvalue()
    string_expected = fixture_data_dir.joinpath("expected/cohesion_score.csv").read_text()
    assert string_actual == string_expected
