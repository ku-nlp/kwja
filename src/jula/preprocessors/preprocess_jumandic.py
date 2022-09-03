import logging
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

from jinf import Jinf
from jumandic import JumanDIC

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", level=logging.DEBUG)


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
        help="path to directory to save data",
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
    dic = JumanDIC(args.input_dir)
    dic.export(str(outdir / "jumandic.dic"))

    # pos:subpos:conjtype:conjform -> surf -> list of lemmas
    ambig_surf2lemmas: dict[str, dict[str, list[str]]] = {}
    jinf = Jinf()
    for entry in dic:
        for ambig_surf_spec in ambig_surf_specs:
            if entry.conjtype == ambig_surf_spec["conjtype"]:
                conjform = ambig_surf_spec["conjform"]
                signature = f"{entry.pos}:{entry.subpos}:{entry.conjtype}:{conjform}"
                if signature in ambig_surf2lemmas:
                    surf2lemmas = ambig_surf2lemmas[signature]
                else:
                    surf2lemmas = ambig_surf2lemmas[signature] = {}
                for baseform in entry.surf:
                    try:
                        surf = jinf(baseform, entry.conjtype, "基本形", conjform)
                    except IndexError:
                        logger.warning(f"failed to convert {baseform} to {conjform}")
                        pass
                    if surf in surf2lemmas:
                        for baseform2 in surf2lemmas[surf]:
                            if baseform == baseform2:
                                break
                        else:
                            surf2lemmas[surf].append(baseform)
                            logger.info("ambiguity: {}".format(surf2lemmas[surf]))
                    else:
                        surf2lemmas[surf] = [baseform]
    with (outdir / "ambig_surf2lemmas.pkl").open("wb") as f:
        f.write(pickle.dumps(ambig_surf2lemmas))


if __name__ == "__main__":
    main()
