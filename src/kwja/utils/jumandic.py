import json
import struct
from importlib.resources.abc import Traversable
from pathlib import Path

import cdblib

from kwja.utils.logging_util import track

_FIELDS = ("reading", "lemma", "pos", "subpos", "conjtype", "conjform", "semantics")
_ENTRY_FORMAT = "IIHHHHI"  # one id per field


class JumanDic:
    def __init__(self, dic_dir: Path | Traversable) -> None:
        self.jumandic = cdblib.Reader(dic_dir.joinpath("jumandic.db").read_bytes())
        self.jumandic_canon = cdblib.Reader(dic_dir.joinpath("jumandic_canon.db").read_bytes())
        self.grammar_data: dict[str, list[str]] = json.loads(
            dic_dir.joinpath("grammar.json").read_text(encoding="utf-8")
        )

    def lookup_by_norm(self, norm: str) -> list[dict[str, str]]:
        return self._lookup(self.jumandic, norm)

    def lookup_by_canon(self, canon: str) -> list[dict[str, str]]:
        return self._lookup(self.jumandic_canon, canon)

    def _lookup(self, db: cdblib.Reader, key: str) -> list[dict[str, str]]:
        buf = db.get(key.encode("utf-8"))
        if buf is None:
            return []
        return [
            {field: self.grammar_data[f"id2{field}"][value_id] for field, value_id in zip(_FIELDS, ids, strict=True)}
            for ids in struct.iter_unpack(_ENTRY_FORMAT, buf)
        ]

    @classmethod
    def build(cls, out_dir: Path, entries: list[list[str]]) -> None:
        field2value2id: dict[str, dict[str, int]] = {field: {} for field in _FIELDS}
        surf2entries: dict[str, bytes] = {}
        canon2entries: dict[str, bytes] = {}
        for entry in track(entries):
            surf, reading, lemma, pos, subpos, conjtype, conjform, canon, semantics = entry
            ids = [
                field2value2id[field].setdefault(value, len(field2value2id[field]))
                for field, value in zip(
                    _FIELDS, (reading, lemma, pos, subpos, conjtype, conjform, semantics), strict=True
                )
            ]
            packed = struct.pack(_ENTRY_FORMAT, *ids)
            surf2entries[surf] = surf2entries.get(surf, b"") + packed
            canon2entries[canon] = canon2entries.get(canon, b"") + packed

        for file_name, key2entries in (("jumandic.db", surf2entries), ("jumandic_canon.db", canon2entries)):
            with out_dir.joinpath(file_name).open(mode="wb") as f, cdblib.Writer(f) as writer:
                for key, packed_entries in key2entries.items():
                    writer.put(key.encode("utf-8"), packed_entries)

        out_dir.joinpath("grammar.json").write_text(
            json.dumps(
                {f"id2{field}": list(value2id) for field, value2id in field2value2id.items()}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
