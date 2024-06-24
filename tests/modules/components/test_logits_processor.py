import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from kwja.modules.components.logits_processor import (
    SurfForcedDecodingLogitsProcessor,
    TargetProperty,
    get_char2token_items,
    get_reading_candidate_token_ids,
)
from kwja.utils.constants import HALF_SPACE_TOKEN, LEMMA_TOKEN

SPECIAL_TOKENS: List[str] = [f"<extra_id_{idx}>" for idx in range(100)]


def test_get_char2tokens() -> None:
    mt5_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small",
        additional_special_tokens=SPECIAL_TOKENS,
    )
    mt5_char2token_items = get_char2token_items(mt5_tokenizer)
    assert len(mt5_char2token_items) == 19455
    assert mt5_char2token_items["京"] == {
        "京东": 165392,
        "京娱乐": 178804,
        "京都府": 166766,
        "京": 11017,
        "京都市": 209455,
        "京都": 51389,
        "京区": 208641,
        "▁京公网安备": 234066,
    }
    mt5_underscore_tokens: Set[str] = {x for x in mt5_tokenizer.vocab if x.startswith("▁")}
    mt5_non_underscore_tokens: Set[str] = {x for x in mt5_tokenizer.vocab if not x.startswith("▁")}
    assert len(mt5_underscore_tokens) == 56369
    assert len(mt5_non_underscore_tokens) == 193831

    t5_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="retrieva-jp/t5-small-short",
        additional_special_tokens=SPECIAL_TOKENS,
    )
    t5_char2token_items = get_char2token_items(t5_tokenizer)
    assert len(t5_char2token_items) == 4289
    assert t5_char2token_items["京"] == {
        "京都府": 3411,
        "京都府出身": 26029,
        "京橋": 22889,
        "京急": 26569,
        "京都府京都市": 22480,
        "京都": 743,
        "京成": 13372,
        "京浜": 18564,
        "京都帝国大学": 20474,
        "京セラ": 29651,
        "京都大学": 7089,
        "京都府立": 27876,
        "京畿道": 23298,
        "京王": 14545,
        "京極": 21867,
        "京都市": 8841,
        "京": 1351,
        "京阪": 14311,
        "京都市立": 24756,
    }
    t5_underscore_tokens: Set[str] = {x for x in t5_tokenizer.vocab if x.startswith("▁")}
    t5_non_underscore_tokens: Set[str] = {x for x in t5_tokenizer.vocab if not x.startswith("▁")}
    assert len(t5_underscore_tokens) == 531
    assert len(t5_non_underscore_tokens) == 31569


def test_get_target_property(data_dir: Path) -> None:
    model2pretrained_model_name_or_path: Dict[str, str] = {
        "mt5": "google/mt5-small",
        "t5": "retrieva-jp/t5-small-short",
    }
    for model, pretrained_model_name_or_path in model2pretrained_model_name_or_path.items():
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        reading_candidate_token_ids: List[int] = get_reading_candidate_token_ids(tokenizer)
        char2token_items: Dict[str, Dict[str, int]] = get_char2token_items(tokenizer)
        test_case_path: Path = data_dir / "modules" / "permitted_tokens.json"
        with open(test_case_path) as f:
            test_cases = json.load(f)
        for test_case in test_cases.values():
            processor = SurfForcedDecodingLogitsProcessor(
                batch_surfs=[test_case["surfs"]],
                num_beams=1,
                tokenizer=tokenizer,
                char2token_items=char2token_items,
                reading_candidate_token_ids=reading_candidate_token_ids,
            )
            input_ids: List[int] = tokenizer.convert_tokens_to_ids(test_case[model]["input_tokens"])
            target_property: TargetProperty = processor._get_target_property(input_ids)
            assert target_property.surf == (test_case["target_property"] == "surf")
            assert target_property.reading == (test_case["target_property"] == "reading")
            assert target_property.lemma == (test_case["target_property"] == "lemma")
            assert target_property.canon == (test_case["target_property"] == "canon")


@pytest.mark.parametrize(
    ("input_tokens", "surfs", "expected_ungenerated_surf"),
    [
        (["<pad>", "<extra_id_0>"], ["研究", "する"], "研究"),
        (["<pad>", "<extra_id_0>", "研"], ["研究", "する"], "究"),
        (["<pad>", "<extra_id_0>", "研究"], ["研究", "する"], ""),
        (
            [
                "<pad>",
                "<extra_id_0>",
                "研究",
                "<extra_id_1>",
                "けんきゅう",
                "<extra_id_2>",
                "研究",
                "<extra_id_3>",
                "研究",
                "/",
                "けんきゅう",
                "<extra_id_0>",
            ],
            ["研究", "する"],
            "する",
        ),
        (
            [
                "<pad>",
                "<extra_id_0>",
                "研究",
                "<extra_id_1>",
                "けんきゅう",
                "<extra_id_2>",
                "研究",
                "<extra_id_3>",
                "研究",
                "/",
                "けんきゅう",
                "<extra_id_0>",
                "す",
            ],
            ["研究", "する"],
            "る",
        ),
    ],
)
def test_get_ungenerated_surf(input_tokens: List[str], surfs: List[str], expected_ungenerated_surf: str) -> None:
    for pretrained_model_name_or_path in ["google/mt5-small", "retrieva-jp/t5-small-short"]:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        char2token_items = get_char2token_items(tokenizer)
        reading_candidate_token_ids = get_reading_candidate_token_ids(tokenizer)

        processor = SurfForcedDecodingLogitsProcessor(
            batch_surfs=[surfs],
            num_beams=1,
            tokenizer=tokenizer,
            char2token_items=char2token_items,
            reading_candidate_token_ids=reading_candidate_token_ids,
        )
        input_ids: List[int] = tokenizer.convert_tokens_to_ids(input_tokens)
        assert processor._get_ungenerated_surf(input_ids, surfs) == expected_ungenerated_surf


@pytest.mark.parametrize(
    ("surfs", "expected_permitted_tokens"),
    [
        (["研究", "を", "する"], ["研究", "研"]),
        ([HALF_SPACE_TOKEN, "研究", "を", "する"], [HALF_SPACE_TOKEN]),
    ],
)
def test_get_permitted_token_ids(surfs: List[str], expected_permitted_tokens: List[str]) -> None:
    for pretrained_model_name_or_path in ["google/mt5-small", "retrieva-jp/t5-small-short"]:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        char2token_items = get_char2token_items(tokenizer)
        reading_candidate_token_ids = get_reading_candidate_token_ids(tokenizer)

        processor = SurfForcedDecodingLogitsProcessor(
            batch_surfs=[surfs],
            num_beams=2,
            tokenizer=tokenizer,
            char2token_items=char2token_items,
            reading_candidate_token_ids=reading_candidate_token_ids,
        )
        permitted_token_ids: List[int] = processor._get_permitted_token_ids("".join(surfs))
        assert sorted(permitted_token_ids) == sorted(tokenizer.convert_tokens_to_ids(expected_permitted_tokens))


def test_get_mask(data_dir: Path) -> None:
    model2pretrained_model_name_or_path: Dict[str, str] = {
        "mt5": "google/mt5-small",
        "t5": "retrieva-jp/t5-small-short",
    }
    for model, pretrained_model_name_or_path in model2pretrained_model_name_or_path.items():
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        vocab_size: int = len(tokenizer.get_vocab())
        char2token_items = get_char2token_items(tokenizer)
        reading_candidate_token_ids = get_reading_candidate_token_ids(tokenizer)
        reading_candidate_tokens: Set[str] = {
            tokenizer.convert_ids_to_tokens(reading_candidate_token_id)
            for reading_candidate_token_id in reading_candidate_token_ids
        }
        all_tokens: Set[str] = set(tokenizer.vocab.keys())

        test_case_path: Path = data_dir / "modules" / "permitted_tokens.json"
        with open(test_case_path) as f:
            test_cases = json.load(f)
        for k, test_case in test_cases.items():
            assert test_case["target_property"] in ["surf", "reading", "lemma", "canon", "init"]
            processor = SurfForcedDecodingLogitsProcessor(
                batch_surfs=[test_case["surfs"]],
                num_beams=1,
                tokenizer=tokenizer,
                char2token_items=char2token_items,
                reading_candidate_token_ids=reading_candidate_token_ids,
            )
            warped_scores: Optional[torch.Tensor] = None
            for idx in range(1, len(test_case[model]["input_tokens"]) + 1):
                input_ids: torch.LongTensor = torch.LongTensor(
                    [tokenizer.convert_tokens_to_ids(test_case[model]["input_tokens"][:idx])]
                )
                orig_scores: torch.Tensor = torch.full((1, vocab_size), 0.5).float()
                warped_scores = processor(input_ids, orig_scores)
                assert warped_scores is not None
                assert warped_scores.shape == orig_scores.shape
            assert warped_scores is not None
            permitted_tokens: Set[str] = set()
            for token_id, score in enumerate(warped_scores.tolist()[0]):
                if score == 0.5:
                    permitted_tokens.add(tokenizer.convert_ids_to_tokens(token_id))

            if test_case[model]["permitted_tokens"] == "reading_candidates":
                expected_permitted_tokens: Set[str] = reading_candidate_tokens
                expected_permitted_tokens.add(LEMMA_TOKEN)
            elif len(test_case[model]["permitted_tokens"]) == 0:
                expected_permitted_tokens = copy.deepcopy(all_tokens)
            else:
                expected_permitted_tokens = set(test_case[model]["permitted_tokens"])

            if "prohibited_tokens" in test_case[model]:
                expected_permitted_tokens -= set(test_case[model]["prohibited_tokens"])
            assert sorted(list(permitted_tokens)) == sorted(list(expected_permitted_tokens))
