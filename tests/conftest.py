import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from omegaconf import ListConfig
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.metrics.cohesion_scorer import Scorer

os.environ["DATA_DIR"] = ""
base_path = Path(__file__).parent.parent / "configs" / "base.yaml"
if base_path.exists() is False:
    base_path.symlink_to(base_path.parent / "base_template.yaml")


@pytest.fixture()
def fixture_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture()
def typo_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        "ku-nlp/deberta-v2-tiny-japanese-char-wwm",
        do_word_tokenize=False,
        additional_special_tokens=["<k>", "<d>", "<_>", "<dummy>"],
    )


@pytest.fixture()
def seq2seq_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        "google/mt5-small",
        additional_special_tokens=[f"<extra_id_{idx}>" for idx in range(100)],
    )


@pytest.fixture()
def char_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese-char-wwm", do_word_tokenize=False)


@pytest.fixture()
def word_tokenizer(special_tokens: List[str]) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese", additional_special_tokens=special_tokens)


@pytest.fixture()
def cohesion_tasks() -> List[str]:
    return ["pas_analysis", "bridging_reference_resolution", "coreference_resolution"]


@pytest.fixture()
def exophora_referents() -> List[str]:
    return ["著者", "読者", "不特定:人", "不特定:物"]


@pytest.fixture()
def pas_cases() -> List[str]:
    return ["ガ", "ヲ", "ニ", "ガ２"]


@pytest.fixture()
def br_cases() -> List[str]:
    return ["ノ"]


@pytest.fixture()
def special_tokens(exophora_referents: List[str]) -> List[str]:
    return [f"[{e}]" for e in exophora_referents] + ["[NULL]", "[NA]", "[ROOT]"]


@pytest.fixture()
def dataset_kwargs(
    cohesion_tasks: List[str],
    exophora_referents: List[str],
    pas_cases: List[str],
    br_cases: List[str],
    special_tokens: List[str],
) -> Dict[str, Any]:
    return {
        "cohesion_tasks": ListConfig(cohesion_tasks),
        "exophora_referents": ListConfig(exophora_referents),
        "restrict_cohesion_target": True,
        "pas_cases": ListConfig(pas_cases),
        "br_cases": ListConfig(br_cases),
        "special_tokens": ListConfig(special_tokens),
    }


@pytest.fixture()
def fixture_scorer(fixture_data_dir: Path, exophora_referents: List[str]):
    predicted_documents = [Document.from_knp(path.read_text()) for path in fixture_data_dir.glob("system/*.knp")]
    gold_documents = [Document.from_knp(path.read_text()) for path in fixture_data_dir.glob("gold/*.knp")]

    scorer = Scorer(
        predicted_documents,
        gold_documents,
        exophora_referents=[ExophoraReferent(e) for e in exophora_referents],
        pas_cases=["ガ", "ヲ"],
        pas_target="all",
        bridging=True,
        coreference=True,
    )
    return scorer
