import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest
import torch
from rhoknp import Sentence
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.datamodule.datasets.seq2seq import Seq2SeqFormatter
from kwja.modules.components.logits_processor import (
    ForcedLogitsProcessor,
    TargetMorpheme,
    get_char2tokens,
    get_reading_candidates,
)
from kwja.utils.constants import FULL_SPACE_TOKEN, HALF_SPACE_TOKEN1, HALF_SPACE_TOKEN2, LEMMA_TOKEN, TRIPLE_DOT_TOKEN

SPECIAL_TOKENS: List[str] = [f"<extra_id_{idx}>" for idx in range(100)]


def test_get_char2tokens() -> None:
    mt5_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-small",
        additional_special_tokens=SPECIAL_TOKENS,
    )
    mt5_char2tokens = get_char2tokens(mt5_tokenizer)
    assert len(mt5_char2tokens) == 19455
    assert mt5_char2tokens["京"] == {
        "京东": 165392,
        "京娱乐": 178804,
        "京都府": 166766,
        "京": 11017,
        "京都市": 209455,
        "京都": 51389,
        "京区": 208641,
        "▁京公网安备": 234066,
    }
    mt5_underscore_tokens: Set[str] = set([x for x in mt5_tokenizer.get_vocab() if x.startswith("▁")])
    mt5_non_underscore_tokens: Set[str] = set([x for x in mt5_tokenizer.get_vocab() if not x.startswith("▁")])
    assert len(mt5_underscore_tokens) == 56369
    assert len(mt5_non_underscore_tokens) == 193831

    t5_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="retrieva-jp/t5-small-short",
        additional_special_tokens=SPECIAL_TOKENS,
    )
    t5_char2tokens = get_char2tokens(t5_tokenizer)
    assert len(t5_char2tokens) == 4289
    assert t5_char2tokens["京"] == {
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
    t5_underscore_tokens: Set[str] = set([x for x in t5_tokenizer.get_vocab() if x.startswith("▁")])
    t5_non_underscore_tokens: Set[str] = set([x for x in t5_tokenizer.get_vocab() if not x.startswith("▁")])
    assert len(t5_underscore_tokens) == 531
    assert len(t5_non_underscore_tokens) == 31569


def test_get_generated_surfs(data_dir: Path) -> None:
    for pretrained_model_name_or_path in ["google/mt5-small", "retrieva-jp/t5-small-short"]:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        formatter: Seq2SeqFormatter = Seq2SeqFormatter(tokenizer)
        char2tokens = get_char2tokens(tokenizer)
        reading_candidates = get_reading_candidates(tokenizer)

        test_case_dir: Path = data_dir / "modules" / "juman"
        for path in test_case_dir.glob("*.juman"):
            with path.open() as f:
                sentence: Sentence = Sentence.from_jumanpp(f.read())
                processor = ForcedLogitsProcessor(
                    texts=[formatter.sent_to_text(sentence)],
                    num_beams=2,
                    tokenizer=tokenizer,
                    reading_candidates=reading_candidates,
                    char2tokens=char2tokens,
                )

                mrph_lines: List[List[str]] = formatter.sent_to_mrph_lines(sentence)
                tgt_tokens: List[str] = formatter.tokenize(mrph_lines, {})
                tgt_input_ids: torch.Tensor = torch.LongTensor(
                    [tokenizer.convert_tokens_to_ids(tgt_tokens) + [tokenizer.eos_token_id]]
                )

                assert processor.texts[0] == processor.get_generated_surfs(tgt_input_ids)[0].replace(" ", "")


@pytest.mark.parametrize(
    "input_text, permitted_tokens",
    [
        ("研究をする", ["研究", "研"]),
        (f"{FULL_SPACE_TOKEN}研究をする", [FULL_SPACE_TOKEN]),
        (f"{HALF_SPACE_TOKEN1}研究をする", [HALF_SPACE_TOKEN1]),
        (f"{HALF_SPACE_TOKEN2}研究をする", [HALF_SPACE_TOKEN2]),
        (f"{TRIPLE_DOT_TOKEN}研究をする", [TRIPLE_DOT_TOKEN]),
    ],
)
def test_get_banned_token_ids(input_text: str, permitted_tokens: List[str]) -> None:
    for pretrained_model_name_or_path in ["google/mt5-small", "retrieva-jp/t5-small-short"]:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        reading_candidates: Set[int] = get_reading_candidates(tokenizer)
        char2tokens: Dict[str, Dict[str, int]] = get_char2tokens(tokenizer)
        expected_banned_token_ids: Set[int] = set(tokenizer.get_vocab().values()) - set(
            tokenizer.convert_tokens_to_ids(permitted_tokens)
        )

        processor = ForcedLogitsProcessor(
            texts=[input_text],
            num_beams=2,
            tokenizer=tokenizer,
            reading_candidates=reading_candidates,
            char2tokens=char2tokens,
        )
        banned_token_ids: Set[int] = processor.get_banned_token_ids(input_text)
        assert sorted(list(banned_token_ids)) == sorted(list(expected_banned_token_ids))


def test_get_target_morpheme(data_dir: Path) -> None:
    model2pretrained_model_name_or_path: Dict[str, str] = {
        "mt5": "google/mt5-small",
        "t5": "retrieva-jp/t5-small-short",
    }
    for model, pretrained_model_name_or_path in model2pretrained_model_name_or_path.items():
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        test_case_path: Path = data_dir / "modules" / "permitted_tokens.json"
        with open(test_case_path) as f:
            test_cases = json.load(f)
        for test_id, test_case in test_cases.items():
            reading_candidates: Set[int] = get_reading_candidates(tokenizer)
            char2tokens: Dict[str, Dict[str, int]] = get_char2tokens(tokenizer)
            processor = ForcedLogitsProcessor(
                texts=[test_case["text"]],
                num_beams=1,
                tokenizer=tokenizer,
                reading_candidates=reading_candidates,
                char2tokens=char2tokens,
            )
            input_ids: List[int] = tokenizer.convert_tokens_to_ids(test_case[model]["input_tokens"])
            target_morpheme: TargetMorpheme = processor.get_target_morpheme(input_ids)
            assert target_morpheme.surf == (test_case["target_morpheme"] == "surf")
            assert target_morpheme.reading == (test_case["target_morpheme"] == "reading")
            assert target_morpheme.lemma == (test_case["target_morpheme"] == "lemma")
            assert target_morpheme.canon == (test_case["target_morpheme"] == "canon")


def test_get_mask(data_dir: Path) -> None:
    model2pretrained_model_name_or_path: Dict[str, str] = {
        "mt5": "google/mt5-small",
        "t5": "retrieva-jp/t5-small-short",
    }
    for model, pretrained_model_name_or_path in model2pretrained_model_name_or_path.items():
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        vocab_size: int = len(tokenizer.get_vocab())
        reading_candidates: Set[int] = get_reading_candidates(tokenizer)
        reading_candidate_tokens: Set[str] = set([tokenizer.convert_ids_to_tokens([x])[0] for x in reading_candidates])
        char2tokens = get_char2tokens(tokenizer)
        all_tokens: Set[str] = set(tokenizer.get_vocab().keys())

        test_case_path: Path = data_dir / "modules" / "permitted_tokens.json"
        with open(test_case_path) as f:
            test_cases = json.load(f)
        for test_id, test_case in test_cases.items():
            assert test_case["target_morpheme"] in ["surf", "reading", "lemma", "canon", "init"]
            processor = ForcedLogitsProcessor(
                texts=[test_case["text"]],
                num_beams=1,
                tokenizer=tokenizer,
                reading_candidates=reading_candidates,
                char2tokens=char2tokens,
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
            elif not test_case[model]["permitted_tokens"]:
                expected_permitted_tokens = copy.deepcopy(all_tokens)
            else:
                expected_permitted_tokens = set(test_case[model]["permitted_tokens"])

            if "banned_tokens" in test_case[model]:
                expected_permitted_tokens -= set(test_case[model]["banned_tokens"])
            assert sorted(list(permitted_tokens)) == sorted(list(expected_permitted_tokens))
