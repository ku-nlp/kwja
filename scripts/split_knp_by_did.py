import argparse
import logging
import pathlib

from kyoto_reader import KyotoReader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Split a KNP file according to document IDs")
    parser.add_argument("--input_file", help="Input file")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--ext", default=".knp", help="Output file extension")
    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file)
    reader = KyotoReader(input_file, did_from_sid=True)
    output_dir = pathlib.Path(args.output_dir)

    for doc_id in tqdm(reader.doc_ids):
        knp = reader.get_knp(doc_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_dir.joinpath(f"{doc_id}{args.ext}").open(mode="wt") as f:
            f.write(knp)


if __name__ == "__main__":
    main()
