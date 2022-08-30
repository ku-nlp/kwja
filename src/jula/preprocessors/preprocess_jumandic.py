import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

from jinf import Jinf
from jumandic import JumanDIC


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="path to JumanDIC directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./data",
        help="path to directory to save",
    )
    args = parser.parse_args()
    if not Path(args.input_dir).is_dir():
        sys.stderr.write("--input-dir must be an existing JumanDIC dir\n")
        exit(1)
    outdir = Path(args.output_dir)
    if outdir.exists():
        if not outdir.is_dir():
            sys.stderr.write("--output-dir must be a directory\n")
            exit(1)
    else:
        outdir.mkdir(parents=True)

    ebase2bases: dict[str, list[str]] = {}
    jinf = Jinf()
    jumandic = JumanDIC(args.input_dir)
    for entry in jumandic:
        if entry.conjtype in ("イ形容詞アウオ段", "イ形容詞イ段", "イ形容詞イ段特殊"):
            for surf in entry.surf:
                try:
                    ebase_form = jinf(surf, entry.conjtype, "基本形", "エ基本形")
                except IndexError:
                    # いい
                    pass
                if ebase_form in ebase2bases:
                    for surf2 in ebase2bases[ebase_form]:
                        if surf == surf2:
                            break
                    else:
                        ebase2bases[ebase_form].append(surf)
                        print(ebase2bases[ebase_form])
                else:
                    ebase2bases[ebase_form] = [surf]
    with (outdir / "ebase2bases.pkl").open("wb") as f:
        f.write(pickle.dumps(ebase2bases))


if __name__ == "__main__":
    main()
