import logging
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Dict

from rhoknp import Document
from transformers import AutoTokenizer

from kwja.utils.kanjidic import KanjiDic
from kwja.utils.reading_prediction import ReadingAligner

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name-or-path", type=str, help="model_name_or_path")
    parser.add_argument("-k", "--kanji-dic", type=str, help="path to kanji dic file")
    parser.add_argument("-i", "--input", type=str, help="path to input directory")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    kanji_dic = KanjiDic(args.kanji_dic)
    reading_aligner = ReadingAligner(tokenizer, kanji_dic)

    reading_counter: Dict[str, int] = Counter()
    for path in Path(args.input).glob("**/*.knp"):
        logger.info(f"processing {path}")
        with path.open() as f:
            document = Document.from_knp(f.read())
        try:
            for reading in reading_aligner.align(document.morphemes):
                reading_counter[reading] += 1
        except ValueError:
            logger.warning(f"skip {document.doc_id} for an error")
    for subreading, count in sorted(
        sorted(reading_counter.items(), key=lambda pair: pair[0]), key=lambda pair: pair[1], reverse=True
    ):
        print(f"{subreading}\t{count}")


if __name__ == "__main__":
    main()
