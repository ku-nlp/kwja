import copy
import csv
import logging
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", level=logging.DEBUG)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="path to JumanDIC root directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./data",
        help="path to directory to save data",
    )
    args = parser.parse_args()
    if not Path(args.input_dir).is_dir():
        sys.stderr.write("--input must be an existing JumanDIC root directory\n")
        exit(1)
    input_path = Path(args.input_dir) / "kwja_dic" / "kwja.dic"
    if not input_path.is_file():
        sys.stderr.write(
            "JumanDIC KWJA dictionary not found\nmake sure you have built the KWJA dictionary by calling 'make kwja'\n"
        )
        exit(1)
    outdir = Path(args.output_dir)
    if outdir.exists():
        if not outdir.is_dir():
            sys.stderr.write("--output-dir must be a directory\n")
            exit(1)
    else:
        outdir.mkdir(parents=True)

    # TODO: externalize this
    ambig_surf_specs = [
        {
            "conjtype": "イ形容詞アウオ段",
            "conjform": "エ基本形",
        },
        {
            "conjtype": "イ形容詞イ段",
            "conjform": "エ基本形",
        },
        {
            "conjtype": "イ形容詞イ段特殊",
            "conjform": "エ基本形",
        },
    ]

    with open(str(input_path)) as f:
        dicreader = csv.reader(f)
        rows = list(dicreader)
    entries = []
    for row in tqdm(rows):
        surf, _, _, _, pos, subpos, conjform, conjtype, lemma, reading, repname, sem = row
        semantics = ""
        if repname != "*":
            semantics = f"代表表記:{repname}"
        if sem != "NIL":
            if len(semantics) > 0:
                semantics += " "
            semantics += sem
        entries.append(
            {
                "surf": surf,
                "reading": reading,
                "lemma": lemma,
                "pos": pos,
                "subpos": subpos,
                "conjtype": conjtype,
                "conjform": conjform,
                "semantics": semantics,
            }
        )
    rows = []
    (outdir / "jumandic.dic").unlink(missing_ok=True)
    CachingMiddleware.WRITE_CACHE_SIZE = 1000000
    with TinyDB(str(outdir / "jumandic.dic"), ensure_ascii=False, storage=CachingMiddleware(JSONStorage)) as dic:
        dic.insert_multiple(entries)
    entries = []

    dic = TinyDB(
        str(outdir / "jumandic.dic"), ensure_ascii=False, access_mode="r", storage=CachingMiddleware(JSONStorage)
    )
    # pos:subpos:conjtype:conjform -> surf -> list of lemmas
    ambig_surf2lemmas: dict[str, dict[str, list[str]]] = {}
    for entry in dic:
        for ambig_surf_spec in ambig_surf_specs:
            if entry["conjtype"] == ambig_surf_spec["conjtype"] and entry["conjform"] == ambig_surf_spec["conjform"]:
                signature = f"{entry['pos']}:{entry['subpos']}:{entry['conjtype']}:{entry['conjform']}"
                if signature in ambig_surf2lemmas:
                    surf2lemmas = ambig_surf2lemmas[signature]
                else:
                    surf2lemmas = ambig_surf2lemmas[signature] = {}
                surf = entry["surf"]
                if surf in surf2lemmas:
                    for lemma2 in copy.copy(surf2lemmas[surf]):
                        if entry["lemma"] == lemma2:
                            logger.warning("duplicate: {}".format(surf2lemmas[surf]))
                            break
                        else:
                            surf2lemmas[surf].append(entry["lemma"])
                            logger.info("ambiguity: {}".format(surf2lemmas[surf]))
                else:
                    surf2lemmas[surf] = [entry["lemma"]]
    with (outdir / "ambig_surf2lemmas.pkl").open("wb") as f:
        f.write(pickle.dumps(ambig_surf2lemmas))


if __name__ == "__main__":
    main()
