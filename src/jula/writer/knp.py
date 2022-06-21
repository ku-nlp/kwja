import io
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional, TextIO, Union

from rhoknp import BasePhrase, Document
from rhoknp.rel import ExophoraReferent
from rhoknp.rel.pas import Argument, BaseArgument, Pas, SpecialArgument

from jula.datamodule.datasets.word_dataset import WordDataset
from jula.datamodule.examples import CohesionExample, Task

logger = logging.getLogger(__file__)


class PredictionKNPWriter:
    """A class to write the system output in a KNP format.

    Args:
        dataset (PASDataset): 解析対象のデータセット
    """

    REL_PAT = re.compile(
        r'<rel type="(\S+?)"(?: mode="([^>]+?)")? target="(.*?)"(?: sid="(.*?)" id="(.+?)")?/>'
    )
    TAG_PAT = re.compile(r"^\+ -?\d+\w ?")

    def __init__(
        self,
        dataset: WordDataset,
    ) -> None:
        self.examples: list[CohesionExample] = dataset.examples
        self.cases: list[str] = dataset.cases
        self.tasks: list[Task] = dataset.cohesion_tasks
        self.relations: list[str] = (
            self.cases * (Task.PAS_ANALYSIS in self.tasks)
            + ["ノ"] * (Task.BRIDGING in self.tasks)
            + ["="] * (Task.COREFERENCE in self.tasks)
        )
        self.exophora_referents: list[ExophoraReferent] = dataset.exophora_referents
        self.specials: list[str] = dataset.special_tokens
        self.documents: list[Document] = dataset.documents
        self.kc: bool = False

    def write(
        self,
        predictions: dict[int, list[list[int]]],
        destination: Union[Path, TextIO, None] = None,
        skip_untagged: bool = True,
        add_pas_tag: bool = True,
    ) -> list[Document]:
        """Write final predictions to files.

        Args:
            predictions (dict[int, list[list[int]]]): example_idをkeyとする基本句単位モデル出力
            destination (Union[Path, TextIO, None]): 解析済み文書の出力先 (default: None)
            skip_untagged (bool): 解析に失敗した文書を出力しないかどうか (default: True)
            add_pas_tag (bool): 解析結果に<述語項構造 >タグを付与するかどうか (default: True)
        Returns:
            list[Document]: 解析済み文書
        """

        if isinstance(destination, Path):
            logger.info(f"Writing predictions to: {destination}")
            destination.mkdir(exist_ok=True)
        elif not (destination is None or isinstance(destination, io.TextIOBase)):
            logger.warning("invalid output destination")

        did2prediction: dict[str, list] = {
            self.examples[eid].doc_id: pred for eid, pred in predictions.items()
        }
        did2knps: dict[str, list[str]] = defaultdict(list)
        for document in self.documents:
            did = document.doc_id
            input_knp_lines: list[str] = document.knp_string.strip().splitlines()
            if prediction := did2prediction.get(did):  # (phrase, rel)
                output_knp_lines = self._rewrite_rel(
                    input_knp_lines,
                    prediction,
                    document,
                )  # overtを抽出するためこれはreparse後に格解析したものがいい
            else:
                if skip_untagged is True:
                    continue
                assert all("<rel " not in line for line in input_knp_lines)
                output_knp_lines = input_knp_lines

            knp_strings: list[str] = []  # list of knp_string of one sentence
            buff = ""
            for knp_line in output_knp_lines:
                buff += knp_line + "\n"
                if knp_line.strip() == "EOS":
                    knp_strings.append(buff)
                    buff = ""
            if self.kc is True:
                # merge documents
                orig_did, idx = did.split("-")
                if idx == "00":
                    did2knps[orig_did] += knp_strings
                else:
                    did2knps[orig_did].append(knp_strings[-1])
            else:
                did2knps[did] = knp_strings

        documents_pred: list[Document] = []  # kc については元通り結合された文書のリスト
        for did, knp_strings in did2knps.items():
            document_pred = Document.from_knp("".join(knp_strings))
            document_pred.doc_id = did
            documents_pred.append(document_pred)
            if destination is None:
                continue
            output_knp_lines = "".join(knp_strings).strip().split("\n")
            if add_pas_tag is True:
                output_knp_lines = self._add_pas_tag(output_knp_lines, document_pred)
            output_string = "\n".join(output_knp_lines) + "\n"
            if isinstance(destination, Path):
                destination.joinpath(f"{did}.knp").write_text(output_string)
            elif isinstance(destination, io.TextIOBase):
                destination.write(output_string)

        return documents_pred

    def _rewrite_rel(
        self,
        knp_lines: list[str],
        prediction: list[list[int]],  # (phrase, rel)
        document: Document,  # <格解析>付き
    ) -> list[str]:
        base_phrases: list[BasePhrase] = document.base_phrases

        output_knp_lines = []
        dtid = 0
        sent_idx = 0
        for line in knp_lines:
            if not line.startswith("+ "):
                output_knp_lines.append(line)
                if line == "EOS":
                    sent_idx += 1
                continue

            assert "<rel " not in line
            if match := self.TAG_PAT.match(line):
                rel_string = self._rel_string(
                    prediction[dtid],
                    base_phrases,
                )
                rel_idx = match.end()
                output_knp_lines.append(line[:rel_idx] + rel_string + line[rel_idx:])
            else:
                logger.warning(f"invalid format line: {line}")
                output_knp_lines.append(line)

            dtid += 1

        return output_knp_lines

    def _rel_string(
        self,
        prediction: list[int],  # (rel)
        bp_list: list[BasePhrase],
    ) -> str:
        rels: list[RelTag] = []
        assert len(self.relations) == len(prediction)
        for relation, pred in zip(self.relations, prediction):
            if pred < 0:
                continue  # non-target phrase
            if 0 <= pred < len(bp_list):
                # normal
                prediction_bp: BasePhrase = bp_list[pred]
                rels.append(
                    RelTag(
                        relation,
                        prediction_bp.head.text,
                        prediction_bp.sentence.sid,
                        prediction_bp.index,
                    )
                )
            elif 0 <= pred - len(bp_list) < len(self.specials):
                # special
                special_arg = self.specials[pred - len(bp_list)]
                if special_arg in [
                    str(e) for e in self.exophora_referents
                ]:  # exclude [NULL] and [NA]
                    rels.append(RelTag(relation, special_arg, None, None))
            else:
                raise ValueError(f"invalid pred index: {pred}")

        return "".join(rel.to_string() for rel in rels)

    def _add_pas_tag(
        self,
        knp_lines: list[str],
        document: Document,
    ) -> list[str]:
        dtid2pas = {
            pas.predicate.base_phrase.global_index: pas for pas in document.pas_list()
        }
        dtid = 0
        output_knp_lines = []
        for line in knp_lines:
            if not line.startswith("+ "):
                output_knp_lines.append(line)
                continue
            if dtid in dtid2pas:
                pas_string = self._pas_string(dtid2pas[dtid], "dummy:dummy", document)
                output_knp_lines.append(line + pas_string)
            else:
                output_knp_lines.append(line)

            dtid += 1

        return output_knp_lines

    def _pas_string(
        self,
        pas: Pas,
        cfid: str,
        document: Document,
    ) -> str:
        sid2index: dict[str, int] = {
            sent.sid: i for i, sent in enumerate(document.sentences)
        }
        # dtype2caseflag = {'overt': 'C', 'dep': 'N', 'intra': 'O', 'inter': 'O', 'exo': 'E'}
        case_elements = []
        for case in self.cases + ["ノ"] * (Task.BRIDGING in self.tasks):
            items = ["-"] * 6
            items[0] = case
            args = pas.arguments[case]
            if args:
                arg: BaseArgument = args[0]
                items[1] = str(arg.type)  # フラグ (C/N/O/D/E/U)
                items[2] = str(arg)  # 見出し
                if isinstance(arg, Argument):
                    items[3] = str(
                        sid2index[pas.sid] - sid2index[arg.base_phrase.sentence.sid]
                    )  # N文前
                    items[4] = str(arg.base_phrase.index)  # tag id
                    items[5] = str(list(arg.base_phrase.entities)[0].eid)  # Entity ID
                else:
                    assert isinstance(arg, SpecialArgument)
                    items[3] = str(-1)
                    items[4] = str(-1)
                    items[5] = str(arg.eid)  # Entity ID
            else:
                items[1] = "U"
            case_elements.append("/".join(items))
        return f"<述語項構造:{cfid}:{';'.join(case_elements)}>"


class RelTag(NamedTuple):
    type_: str
    target: str
    sid: Optional[str]
    tid: Optional[int]

    def to_string(self) -> str:
        string = f'<rel type="{self.type_}" target="{self.target}"'
        if self.sid is not None:
            string += f' sid="{self.sid}" id="{self.tid}"'
        string += "/>"
        return string
