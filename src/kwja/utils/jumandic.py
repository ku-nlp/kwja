import json
import struct
from pathlib import Path
from typing import Dict, List

import cdblib
from tqdm import tqdm


class JumanDic:
    def __init__(self, dicdir: Path) -> None:
        self.dic_path = Path(dicdir)
        with open(str(self.dic_path / "jumandic.db"), "rb") as f:
            data = f.read()
        self.jumandic = cdblib.Reader(data)
        with open(str(self.dic_path / "grammar.json"), "r") as f:
            data2 = json.loads(f.read())
        self.id2reading = data2["id2reading"]
        self.id2lemma = data2["id2lemma"]
        self.id2pos = data2["id2pos"]
        self.id2subpos = data2["id2subpos"]
        self.id2conjtype = data2["id2conjtype"]
        self.id2conjform = data2["id2conjform"]
        self.id2semantics = data2["id2semantics"]

    def lookup(self, surf: str) -> List[Dict[str, str]]:
        buf = self.jumandic.get(surf.encode("utf-8"))
        if buf is None:
            return []
        matches = []
        for reading_id, lemma_id, pos_id, subpos_id, conjtype_id, conjform_id, semantics_id in struct.iter_unpack(
            "IIHHHHI", buf
        ):
            matches.append(
                {
                    "surf": surf,
                    "reading": self.id2reading[reading_id],
                    "lemma": self.id2lemma[lemma_id],
                    "pos": self.id2pos[pos_id],
                    "subpos": self.id2subpos[subpos_id],
                    "conjtype": self.id2conjtype[conjtype_id],
                    "conjform": self.id2conjform[conjform_id],
                    "semantics": self.id2semantics[semantics_id],
                }
            )
        return matches

    @classmethod
    def build(cls, outdir: Path, entries: List[List[str]]) -> None:
        surf2entries: Dict[str, bytes] = {}
        reading2id: Dict[str, int] = {}
        lemma2id: Dict[str, int] = {}
        pos2id: Dict[str, int] = {}
        subpos2id: Dict[str, int] = {}
        conjtype2id: Dict[str, int] = {}
        conjform2id: Dict[str, int] = {}
        semantics2id: Dict[str, int] = {}

        def _get_counter(d, k) -> int:
            if k in d:
                return d[k]
            else:
                rv = len(d)
                d[k] = rv
                return rv

        def _build_reverse_lookup(d) -> List[str]:
            rv: List[str] = [""] * len(d)
            for k, v in d.items():
                rv[v] = k
            return rv

        for entry in tqdm(entries):
            surf, reading, lemma, pos, subpos, conjtype, conjform, semantics = entry
            reading_id = _get_counter(reading2id, reading)
            lemma_id = _get_counter(lemma2id, lemma)
            pos_id = _get_counter(pos2id, pos)
            subpos_id = _get_counter(subpos2id, subpos)
            conjtype_id = _get_counter(conjtype2id, conjtype)
            conjform_id = _get_counter(conjform2id, conjform)
            semantics_id = _get_counter(semantics2id, semantics)
            v = struct.pack(
                "IIHHHHI",
                reading_id,
                lemma_id,
                pos_id,
                subpos_id,
                conjtype_id,
                conjform_id,
                semantics_id,
            )
            if surf in surf2entries:
                surf2entries[surf] += v
            else:
                surf2entries[surf] = v
                (outdir / "jumandic.db").unlink(missing_ok=True)
        with open(str(outdir / "jumandic.db"), "wb") as f:
            with cdblib.Writer(f) as writer:
                for k, v in surf2entries.items():
                    bk = k.encode("utf-8")
                    writer.put(bk, v)
        with open(str(outdir / "grammar.json"), "w") as f2:
            f2.write(
                json.dumps(
                    {
                        "id2reading": _build_reverse_lookup(reading2id),
                        "id2lemma": _build_reverse_lookup(lemma2id),
                        "id2pos": _build_reverse_lookup(pos2id),
                        "id2subpos": _build_reverse_lookup(subpos2id),
                        "id2conjtype": _build_reverse_lookup(conjtype2id),
                        "id2conjform": _build_reverse_lookup(conjform2id),
                        "id2semantics": _build_reverse_lookup(semantics2id),
                    },
                    ensure_ascii=False,
                )
            )
