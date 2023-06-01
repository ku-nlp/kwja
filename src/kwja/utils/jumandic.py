import json
import struct
from pathlib import Path
from typing import Dict, List

import cdblib

from kwja.utils.logging_util import track


class JumanDic:
    def __init__(self, dic_dir: Path) -> None:
        self.jumandic = cdblib.Reader(dic_dir.joinpath("jumandic.db").read_bytes())
        self.jumandic_canon = cdblib.Reader(dic_dir.joinpath("jumandic_canon.db").read_bytes())
        self.grammar_data: Dict[str, List[str]] = json.loads(dic_dir.joinpath("grammar.json").read_text())

    def lookup_by_norm(self, surf: str) -> List[Dict[str, str]]:
        buf = self.jumandic.get(surf.encode("utf-8"))
        if buf is None:
            return []
        matches: List[Dict[str, str]] = []
        for reading_id, lemma_id, pos_id, subpos_id, conjtype_id, conjform_id, semantics_id in struct.iter_unpack(
            "IIHHHHI", buf
        ):
            matches.append(
                {
                    "surf": surf,
                    "reading": self.grammar_data["id2reading"][reading_id],
                    "lemma": self.grammar_data["id2lemma"][lemma_id],
                    "pos": self.grammar_data["id2pos"][pos_id],
                    "subpos": self.grammar_data["id2subpos"][subpos_id],
                    "conjtype": self.grammar_data["id2conjtype"][conjtype_id],
                    "conjform": self.grammar_data["id2conjform"][conjform_id],
                    "semantics": self.grammar_data["id2semantics"][semantics_id],
                }
            )
        return matches

    def lookup_by_canon(self, canon: str) -> List[Dict[str, str]]:
        buf = self.jumandic_canon.get(canon.encode("utf-8"))
        if buf is None:
            return []
        matches: List[Dict[str, str]] = []
        for reading_id, lemma_id, pos_id, subpos_id, conjtype_id, conjform_id, semantics_id in struct.iter_unpack(
            "IIHHHHI", buf
        ):
            matches.append(
                {
                    "canon": canon,
                    "reading": self.grammar_data["id2reading"][reading_id],
                    "lemma": self.grammar_data["id2lemma"][lemma_id],
                    "pos": self.grammar_data["id2pos"][pos_id],
                    "subpos": self.grammar_data["id2subpos"][subpos_id],
                    "conjtype": self.grammar_data["id2conjtype"][conjtype_id],
                    "conjform": self.grammar_data["id2conjform"][conjform_id],
                    "semantics": self.grammar_data["id2semantics"][semantics_id],
                }
            )
        return matches

    @classmethod
    def build(cls, out_dir: Path, entries: List[List[str]]) -> None:
        surf2entries: Dict[str, bytes] = {}
        canon2entries: Dict[str, bytes] = {}

        reading2id: Dict[str, int] = {}
        lemma2id: Dict[str, int] = {}
        pos2id: Dict[str, int] = {}
        subpos2id: Dict[str, int] = {}
        conjtype2id: Dict[str, int] = {}
        conjform2id: Dict[str, int] = {}
        semantics2id: Dict[str, int] = {}

        def _get_counter(dic: Dict[str, int], key: str) -> int:
            if key in dic:
                return dic[key]
            else:
                rv = len(dic)
                dic[key] = rv
                return rv

        def _build_reverse_lookup(dic) -> List[str]:
            rv: List[str] = [""] * len(dic)
            for k, v in dic.items():
                rv[v] = k
            return rv

        for entry in track(entries):
            surf, reading, lemma, pos, subpos, conjtype, conjform, canon, semantics = entry
            reading_id: int = _get_counter(reading2id, reading)
            lemma_id: int = _get_counter(lemma2id, lemma)
            pos_id: int = _get_counter(pos2id, pos)
            subpos_id: int = _get_counter(subpos2id, subpos)
            conjtype_id: int = _get_counter(conjtype2id, conjtype)
            conjform_id: int = _get_counter(conjform2id, conjform)
            semantics_id: int = _get_counter(semantics2id, semantics)
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
            surf2entries[surf] = surf2entries.get(surf, b"") + v
            canon2entries[canon] = canon2entries.get(canon, b"") + v

        with out_dir.joinpath("jumandic.db").open(mode="wb") as f1:
            with cdblib.Writer(f1) as writer1:
                for key_surf, value_entries in surf2entries.items():
                    writer1.put(key_surf.encode("utf-8"), value_entries)

        with out_dir.joinpath("jumandic_canon.db").open(mode="wb") as f2:
            with cdblib.Writer(f2) as writer2:
                for key_canon, value_entries in canon2entries.items():
                    writer2.put(key_canon.encode("utf-8"), value_entries)

        out_dir.joinpath("grammar.json").write_text(
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
