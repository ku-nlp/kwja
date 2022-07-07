import json
from pathlib import Path

from rhoknp import Document
from rhoknp.rel import ExophoraReferent

from jula.evaluators.cohesion_scorer import Measure, Scorer


def test_scorer(fixture_data_dir: Path):
    documents_pred = [
        Document.from_knp(path.read_text())
        for path in fixture_data_dir.glob("system/*.knp")
    ]
    documents_gold = [
        Document.from_knp(path.read_text())
        for path in fixture_data_dir.glob("gold/*.knp")
    ]

    with fixture_data_dir.joinpath("expected/cohesion_score.json").open() as f:
        expected_scores = json.load(f)

    cases = ["ガ", "ヲ"]
    scorer = Scorer(
        documents_pred,
        documents_gold,
        target_cases=cases,
        exophora_referents=[
            ExophoraReferent(e) for e in ("著者", "読者", "不特定:人", "不特定:物")
        ],
        coreference=True,
        bridging=True,
        pas_target="all",
    )

    result = scorer.run().to_dict()
    print(result)
    for case in cases:
        case_result = result[case]
        for anal in Scorer.DEPTYPE2ANALYSIS.values():
            expected: dict = expected_scores[case][anal]
            actual: Measure = case_result[anal]
            assert expected["denom_pred"] == actual.denom_pred
            assert expected["denom_gold"] == actual.denom_gold
            assert expected["correct"] == actual.correct
