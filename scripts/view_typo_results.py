import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

from Levenshtein import opcodes


def apply_ops(pre_text: str, kdrs: List[str], inss: List[str]) -> str:
    post_text = ""
    for i, c in enumerate(pre_text):
        if inss[i] != "_":
            post_text += inss[i][2:]
        if kdrs[i] == "K":
            post_text += c
        elif kdrs[i] == "D":
            pass
        elif len(kdrs[i]) > 2 and kdrs[i][0:2] == "R:":
            post_text += kdrs[i][2:]
        else:
            raise ValueError("unsupported op")
    if inss[-1] != "_":
        post_text += inss[-1][2:]
    return post_text


def get_diffs(before: str, after: str) -> List[Tuple[str, str]]:
    before_split: List[str] = list(before)
    after_split: List[str] = list(after)

    char2id: Dict[str, str] = dict()
    id2char: Dict[str, str] = dict()
    id_ = 0
    for char in before_split + after_split:
        if char not in char2id:
            char2id[char] = chr(id_)
            id2char[chr(id_)] = char
            id_ += 1

    before_id_string: str = "".join([char2id[i] for i in before_split])
    after_id_string: str = "".join([char2id[i] for i in after_split])

    diff_list: List[Tuple[str, str]] = []
    for operation, bb, be, ab, ae in opcodes(before_id_string, after_id_string):
        if operation == "insert":
            for char_id in after_id_string[ab:ae]:
                diff_list.append(("", id2char[char_id]))
        if operation == "delete":
            for char_id in before_id_string[bb:be]:
                diff_list.append((id2char[char_id], ""))
        if operation == "replace":
            assert be - bb == ae - ab, (before_split[bb:be], after_split[ab:ae])
            for before_char_id, after_char_id in zip(before_id_string[bb:be], after_id_string[ab:ae]):
                diff_list.append((id2char[before_char_id], id2char[after_char_id]))
    return diff_list


def calc_tp_fp_fn(pre_text: str, ref_post_text: str, pred_post_text: str):
    ref_diffs: List[Tuple[str, str]] = get_diffs(pre_text, ref_post_text)
    pred_diffs: List[Tuple[str, str]] = get_diffs(pre_text, pred_post_text)

    ref_num: int = len(ref_diffs)
    tp: int = 0
    for diff in pred_diffs:
        if diff in ref_diffs:
            tp += 1
            ref_diffs.remove(diff)
    fp: int = len(pred_diffs) - tp
    fn: int = ref_num - tp
    assert tp >= 0 and fp >= 0 and fn >= 0, (
        tp,
        fp,
        fn,
        get_diffs(pre_text, ref_post_text),
        get_diffs(pre_text, pred_post_text),
    )

    return tp, fp, fn


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="path to prediction file")
    args = parser.parse_args()

    with Path(args.path).open() as f:
        preds: dict = json.load(f)

    tp_total, fp_total, fn_total = 0, 0, 0
    for example_id, pred in preds.items():
        pre_text: str = pred["input_ids"].replace(" ", "")
        pre_text = pre_text.replace("...", "…")
        ref_post_text: str = apply_ops(pre_text=pre_text, kdrs=pred["kdr_labels"] + ["K"], inss=pred["ins_labels"])
        pred_post_text: str = apply_ops(pre_text=pre_text, kdrs=pred["kdr_preds"] + ["K"], inss=pred["ins_preds"])
        tp, fp, fn = calc_tp_fp_fn(
            pre_text=pre_text,
            ref_post_text=ref_post_text,
            pred_post_text=pred_post_text,
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision: float = tp_total / (tp_total + fp_total)
    recall: float = tp_total / (tp_total + fn_total)
    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
        f05 = (1 + 0.5**2) * precision * recall / (recall + (0.5**2) * precision)
    else:
        f1, f05 = 0, 0

    print("Precision, Recall, F-score, F0.5-score")
    print(
        round(precision * 100, 1),
        "\t",
        round(recall * 100, 1),
        "\t",
        round(f1 * 100, 1),
        "\t",
        round(f05 * 100, 1),
        "\t",
    )


if __name__ == "__main__":
    main()
