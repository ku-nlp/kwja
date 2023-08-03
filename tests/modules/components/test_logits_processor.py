import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest
import torch
from rhoknp import Sentence
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kwja.datamodule.datasets.seq2seq import Seq2SeqFormatter
from kwja.modules.components.logits_processor import ForcedLogitsProcessor, get_char2tokens, get_reading_candidates

SPECIAL_TOKENS: List[str] = [f"<extra_id_{idx}>" for idx in range(100)]


def test_get_char2tokens():
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

    t5_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="retrieva-jp/t5-small-long",
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


def test_get_generated_surfs(data_dir: Path) -> None:
    for pretrained_model_name_or_path in ["google/mt5-small", "retrieva-jp/t5-small-long"]:
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

                seq2seq_format: List[str] = formatter.sent_to_format(sentence)
                tgt_tokens: List[str] = formatter.tokenize(seq2seq_format)
                tgt_input_ids: torch.Tensor = torch.LongTensor(
                    [tokenizer.convert_tokens_to_ids(tgt_tokens) + [tokenizer.eos_token_id]]
                )

                assert processor.texts[0] == processor.get_generated_surfs(tgt_input_ids)[0].replace(" ", "")


@pytest.mark.parametrize("input_text, permitted_tokens", [("研究をする", ["研究", "研"])])
def test_get_permitted_token_ids(input_text: str, permitted_tokens: List[str]) -> None:
    for pretrained_model_name_or_path in ["google/mt5-small", "retrieva-jp/t5-small-long"]:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        reading_candidates = get_reading_candidates(tokenizer)
        char2tokens = get_char2tokens(tokenizer)

        processor = ForcedLogitsProcessor(
            texts=[input_text],
            num_beams=2,
            tokenizer=tokenizer,
            reading_candidates=reading_candidates,
            char2tokens=char2tokens,
        )

        permitted_token_ids: Set[int] = processor.get_permitted_token_ids(input_text)
        sorted_permitted_token_ids: List[int] = sorted(list(permitted_token_ids))
        assert permitted_tokens == tokenizer.convert_ids_to_tokens(sorted_permitted_token_ids)


def test_get_batch_banned_token_ids(data_dir: Path):
    model2pretrained_model_name_or_path: Dict[str, str] = {
        "mt5": "google/mt5-small",
        "t5": "retrieva-jp/t5-small-long",
    }
    for model, pretrained_model_name_or_path in model2pretrained_model_name_or_path.items():
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            additional_special_tokens=SPECIAL_TOKENS,
        )
        vocab_size: int = len(tokenizer.get_vocab())
        reading_candidates: Set[int] = get_reading_candidates(tokenizer)
        reading_candidate_tokens: Set[str] = set([tokenizer.convert_ids_to_tokens([x])[0] for x in reading_candidates])
        char2tokens = get_char2tokens(tokenizer)
        underscore_tokens: Set[str] = set([x for x in tokenizer.get_vocab() if x.startswith("▁")])
        non_underscore_tokens: Set[str] = set([x for x in tokenizer.get_vocab() if not x.startswith("▁")])
        if model == "mt5":
            assert len(underscore_tokens) == 56369
            assert len(non_underscore_tokens) == 193831
        elif model == "t5":
            assert len(underscore_tokens) == 531
            assert len(non_underscore_tokens) == 31569
        else:
            raise ValueError(f"model: {model}")

        test_case_path: Path = data_dir / "modules" / "permitted_tokens.json"
        with open(test_case_path) as f:
            test_cases = json.load(f)
        for test_id, test_case in test_cases.items():
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
                warped_scores = processor(
                    input_ids=input_ids,
                    scores=orig_scores,
                )
                assert warped_scores is not None
                assert warped_scores.shape == orig_scores.shape
            assert warped_scores is not None
            permitted_tokens: Set[str] = set()
            for token_id, score in enumerate(warped_scores.tolist()[0]):
                if score == 0.5:
                    permitted_tokens.add(tokenizer.convert_ids_to_tokens(token_id))

            gold_permitted_tokens: Set[str] = set(test_case[model]["permitted_tokens"])
            if len(permitted_tokens) == vocab_size:
                assert gold_permitted_tokens == set()
            else:
                if test_case.get("is_decoding_reading", False) is True:
                    gold_permitted_tokens = gold_permitted_tokens.union(reading_candidate_tokens)
                assert sorted(list(permitted_tokens)) == sorted(list(gold_permitted_tokens))
