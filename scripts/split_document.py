from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from rhoknp import Document
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


# c.f. https://github.com/nobu-g/cohesion-analysis/blob/main/src/preprocess.py
def split_document(
    input_dir: Path,
    output_dir: Path,
    max_token_length: int,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
):
    for knp_file in tqdm(input_dir.glob("**/*.knp")):
        split = knp_file.parent.name
        with knp_file.open(mode="r") as f:
            document = Document.from_knp(f.read())

        cum_lens = [0]
        for sentence in document.sentences:
            num_tokens = len(tokenizer.tokenize(" ".join(morpheme.surf for morpheme in sentence.morphemes)))
            cum_lens.append(cum_lens[-1] + num_tokens)

        end = 1
        # end を探索
        while end < len(document.sentences) and cum_lens[end + 1] - cum_lens[0] <= max_token_length:
            end += 1

        idx = 0
        while end < len(document.sentences) + 1:
            start = 0
            # start を探索
            while cum_lens[end] - cum_lens[start] > max_token_length:
                start += 1
                if start == end - 1:
                    break

            basename = f"{document.doc_id}-{idx:02}.knp"
            output_dir.joinpath(split).mkdir(parents=True, exist_ok=True)
            with output_dir.joinpath(split).joinpath(basename).open(mode="w") as f:
                f.write("".join(sentence.to_knp() for sentence in document.sentences[start:end]))  # start から end まで書き出し
            idx += 1
            end += 1


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT", help="path to input dir")
    parser.add_argument("OUTPUT", help="path to output dir")
    parser.add_argument(
        "--max-token-length", default=500, type=int, help="truncate document up to the max token length"
    )
    parser.add_argument(
        "--tokenizer",
        default="/mnt/zamia/kodama/jula/model/roberta-base-japanese",
        type=str,
        help="pretrained model name or path",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    split_document(Path(args.INPUT), Path(args.OUTPUT), args.max_token_length, tokenizer)


if __name__ == "__main__":
    main()
