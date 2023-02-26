from pathlib import Path

import pytest
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent

from kwja.metrics.cohesion_scorer import Scorer

here = Path(__file__).parent


@pytest.fixture()
def fixture_data_dir():
    return here / "data"


@pytest.fixture()
def fixture_scorer(fixture_data_dir: Path):
    predicted_documents = [Document.from_knp(path.read_text()) for path in fixture_data_dir.glob("system/*.knp")]
    gold_documents = [Document.from_knp(path.read_text()) for path in fixture_data_dir.glob("gold/*.knp")]

    pas_cases = ["ガ", "ヲ"]
    scorer = Scorer(
        predicted_documents,
        gold_documents,
        exophora_referents=[ExophoraReferent(e) for e in ("著者", "読者", "不特定:人", "不特定:物")],
        pas_cases=pas_cases,
        pas_target="all",
        bridging=True,
        coreference=True,
    )
    return scorer
