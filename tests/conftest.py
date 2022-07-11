from pathlib import Path

import pytest
from rhoknp import Document
from rhoknp.rel import ExophoraReferent

from jula.evaluators.cohesion_scorer import Scorer

here = Path(__file__).parent


@pytest.fixture()
def fixture_data_dir():
    return here / "data"


@pytest.fixture()
def fixture_scorer(fixture_data_dir: Path):
    documents_pred = [
        Document.from_knp(path.read_text())
        for path in fixture_data_dir.glob("system/*.knp")
    ]
    documents_gold = [
        Document.from_knp(path.read_text())
        for path in fixture_data_dir.glob("gold/*.knp")
    ]

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
    return scorer
