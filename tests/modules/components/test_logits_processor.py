import json
from pathlib import Path
from typing import List, Set

import pytest
import torch
from rhoknp import Sentence
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from kwja.datamodule.datasets.seq2seq import get_seq2seq_format
from kwja.modules.components.logits_processor import ForcedSurfLogitsProcessor, get_char2tokens
from kwja.utils.constants import NEW_LINE_TOKEN


def test_get_char2tokens():
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small", additional_special_tokens=["<br>", "<no_read>", "<no_canon>"]
    )

    char2tokens, char2underscore_tokens = get_char2tokens(tokenizer)

    assert len(char2tokens) == 19455
    assert len(char2underscore_tokens) == 1665

    assert char2tokens["京"] == {
        "京东": 165392,
        "京娱乐": 178804,
        "京都府": 166766,
        "京": 11017,
        "京都市": 209455,
        "京都": 51389,
        "京区": 208641,
    }
    assert char2underscore_tokens["京"] == {"▁京公网安备": 234066}


def test_get_generated_surfs(fixture_data_dir: Path) -> None:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small", additional_special_tokens=["<br>", "<no_read>", "<no_canon>"]
    )
    char2tokens, char2underscore_tokens = get_char2tokens(tokenizer)

    test_case_dir: Path = fixture_data_dir / "modules" / "jmn"
    for path in test_case_dir.glob("*.jmn"):
        with path.open() as f:
            sentence: Sentence = Sentence.from_jumanpp(f.read())
            processor = ForcedSurfLogitsProcessor(
                texts=[sentence.text],
                tokenizer=tokenizer,
                char2tokens=char2tokens,
                char2underscore_tokens=char2underscore_tokens,
            )
            tgt_encoding: BatchEncoding = tokenizer(
                get_seq2seq_format(sentence).replace("\n", NEW_LINE_TOKEN),
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=512,
                return_tensors="pt",
            )
            assert processor.texts[0] == processor.get_generated_surfs(tgt_encoding.input_ids)[0]


@pytest.mark.parametrize("input_text, permitted_tokens", [("研究をする", ["研究", "研"])])
def test_get_permitted_token_ids(input_text: str, permitted_tokens: List[str]) -> None:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small", additional_special_tokens=["<br>", "<no_read>", "<no_canon>"]
    )
    char2tokens, char2underscore_tokens = get_char2tokens(tokenizer)

    processor = ForcedSurfLogitsProcessor(
        texts=[input_text],
        tokenizer=tokenizer,
        char2tokens=char2tokens,
        char2underscore_tokens=char2underscore_tokens,
    )

    permitted_token_ids: Set[int] = processor.get_permitted_token_ids(input_text)
    sorted_permitted_token_ids: List[int] = sorted(list(permitted_token_ids))
    assert permitted_tokens == tokenizer.convert_ids_to_tokens(sorted_permitted_token_ids)


@pytest.mark.parametrize("input_text, permitted_underscore_tokens", [("楽天市場", ["▁楽天"])])
def test_get_permitted_underscore_token_ids(input_text: str, permitted_underscore_tokens: List[str]) -> None:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small", additional_special_tokens=["<br>", "<no_read>", "<no_canon>"]
    )
    char2tokens, char2underscore_tokens = get_char2tokens(tokenizer)

    processor = ForcedSurfLogitsProcessor(
        texts=[input_text],
        tokenizer=tokenizer,
        char2tokens=char2tokens,
        char2underscore_tokens=char2underscore_tokens,
    )

    permitted_underscore_token_ids: Set[int] = processor.get_permitted_underscore_token_ids(input_text)
    sorted_permitted_underscore_token_ids: List[int] = sorted(list(permitted_underscore_token_ids))
    assert permitted_underscore_tokens == tokenizer.convert_ids_to_tokens(sorted_permitted_underscore_token_ids)


@pytest.mark.parametrize("input_text, permitted_consecutive_tokens", [("研究をする", ["研究", "研"])])
def test_get_permitted_consecutive_token_ids(input_text: str, permitted_consecutive_tokens: List[str]) -> None:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small", additional_special_tokens=["<br>", "<no_read>", "<no_canon>"]
    )
    char2tokens, char2underscore_tokens = get_char2tokens(tokenizer)

    processor = ForcedSurfLogitsProcessor(
        texts=[input_text],
        tokenizer=tokenizer,
        char2tokens=char2tokens,
        char2underscore_tokens=char2underscore_tokens,
    )

    permitted_consecutive_token_ids: Set[int] = processor.get_permitted_consecutive_token_ids(input_text)
    sorted_permitted_consecutive_token_ids: List[int] = sorted(list(permitted_consecutive_token_ids))
    permitted_tokens: List[str] = tokenizer.convert_ids_to_tokens(sorted_permitted_consecutive_token_ids)
    permitted_tokens_with_underscore: List[str] = []
    permitted_tokens_without_underscore: List[str] = []
    for token in permitted_tokens:
        if token.startswith("▁"):
            permitted_tokens_with_underscore.append(token)
        else:
            permitted_tokens_without_underscore.append(token)
    assert len(permitted_tokens_with_underscore) == 56369
    assert permitted_tokens_without_underscore == permitted_consecutive_tokens


def test_get_batch_banned_token_ids(fixture_data_dir: Path):
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small", additional_special_tokens=["<br>", "<no_read>", "<no_canon>"]
    )
    char2tokens, char2underscore_tokens = get_char2tokens(tokenizer)

    underscore_tokens: List[str] = [x for x in tokenizer.get_vocab() if x.startswith("▁")]
    assert len(underscore_tokens) == 56369

    test_case_path: Path = fixture_data_dir / "modules" / "permitted_tokens.json"
    with open(test_case_path) as f:
        test_cases = json.load(f)
    for test_id, test_case in test_cases.items():
        surf_logits_processor = ForcedSurfLogitsProcessor(
            texts=[test_case["text"]],
            tokenizer=tokenizer,
            char2tokens=char2tokens,
            char2underscore_tokens=char2underscore_tokens,
        )
        input_ids: torch.LongTensor = torch.LongTensor([tokenizer.convert_tokens_to_ids(test_case["input_tokens"])])
        orig_scores: torch.Tensor = torch.full((1, tokenizer.vocab_size), 0.5).float()
        warped_scores: torch.Tensor = surf_logits_processor(
            input_ids=input_ids,
            scores=orig_scores,
        )
        assert warped_scores.shape == orig_scores.shape
        permitted_tokens: List[str] = []
        for token_id, score in enumerate(warped_scores.tolist()[0]):
            if score == 0.5:
                permitted_tokens.append(tokenizer.convert_ids_to_tokens(token_id))
        if len(permitted_tokens) == tokenizer.vocab_size:
            permitted_tokens = []
        if test_case["is_consecutive"] is True:
            gold_permitted_tokens: List[str] = sorted(list(set(test_case["permitted_tokens"] + underscore_tokens)))
        else:
            gold_permitted_tokens = sorted(test_case["permitted_tokens"])
        assert sorted(list(permitted_tokens)) == gold_permitted_tokens
