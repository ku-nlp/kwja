from typing import Dict, List, Literal, Tuple

from transformers import PreTrainedTokenizerBase

from kwja.utils.constants import TOKEN2TYPO_CORR_OP_TAG


def get_maps(tokenizer: PreTrainedTokenizerBase, extended_vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    token2token_id = tokenizer.get_vocab()
    with open(extended_vocab_path, mode="r") as f:
        for line in f:
            if line := line.strip():
                token2token_id[line] = len(token2token_id.keys())
    token_id2token = {v: k for k, v in token2token_id.items()}
    return token2token_id, token_id2token


def convert_predictions_into_typo_corr_op_tags(
    predictions: List[int],
    probabilities: List[float],
    prefix: Literal["R", "I"],
    confidence_threshold: float,
    token2token_id: Dict[str, int],
    token_id2token: Dict[int, str],
) -> List[str]:
    typo_corr_op_tags: List[str] = []
    for token_id, probability in zip(predictions, probabilities):
        # do not edit if the probability (replace, delete, and insert) is less than "confidence_threshold"
        if probability < confidence_threshold:
            token_id = token2token_id["<k>"] if prefix == "R" else token2token_id["<_>"]
        token: str = token_id2token[token_id]
        typo_corr_op_tag = TOKEN2TYPO_CORR_OP_TAG.get(token, f"{prefix}:{token}")
        typo_corr_op_tags.append(typo_corr_op_tag)
    return typo_corr_op_tags


def apply_edit_operations(pre_text: str, kdr_tags: List[str], ins_tags: List[str]) -> str:
    assert len(pre_text) + 1 == len(kdr_tags) + 1 == len(ins_tags)
    post_text = ""
    for i, char in enumerate(pre_text):
        if ins_tags[i].startswith("I:"):
            post_text += ins_tags[i][2:]  # remove prefix "I:"

        if kdr_tags[i] in {"K", "_"}:  # "_"はinsert用のタグなのでkdr_tags[i] == "_"となることは滅多にない
            post_text += char
        elif kdr_tags[i] == "D":
            pass
        elif kdr_tags[i].startswith("R:"):
            post_text += kdr_tags[i][2:]  # remove prefix "R:"
    if ins_tags[-1].startswith("I:"):
        post_text += ins_tags[-1][2:]  # remove prefix "I:"
    return post_text
