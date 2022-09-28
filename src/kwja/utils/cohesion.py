import io
import logging
from collections import defaultdict
from pathlib import Path
from typing import TextIO, Union

from rhoknp import BasePhrase, Document, Sentence
from rhoknp.cohesion import Argument, EndophoraArgument, ExophoraArgument, ExophoraReferent, Pas, RelTag, RelTagList

from kwja.datamodule.datasets import WordDataset
from kwja.datamodule.examples import CohesionExample, CohesionTask
from kwja.utils.sub_document import extract_target_sentences, to_orig_doc_id

logger = logging.getLogger(__name__)


class CohesionKNPWriter:
    """A class to write the system output in a KNP format.

    Args:
        dataset (WordDataset): 解析対象のデータセット
    """

    def __init__(self, dataset: WordDataset) -> None:
        self.cases: list[str] = dataset.pas_cases
        self.tasks: list[CohesionTask] = dataset.cohesion_tasks
        self.rel_types: list[str] = dataset.cohesion_rel_types
        self.exophora_referents: list[ExophoraReferent] = dataset.exophora_referents
        self.specials: list[str] = dataset.special_tokens
        self.documents: list[Document] = [self.reconstruct(doc) for doc in dataset.documents]
        self.examples: list[CohesionExample] = [e.cohesion_example for e in dataset.examples]
        self.kc: bool = False

    @staticmethod
    def reconstruct(doc):
        doc_id = doc.doc_id
        reconstructed = Document.from_knp(doc.to_knp())
        reconstructed.doc_id = doc_id
        return reconstructed

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

        did2prediction: dict[str, list] = {self.examples[eid].doc_id: pred for eid, pred in predictions.items()}
        orig_did_to_sentences: dict[str, list[Sentence]] = defaultdict(list)
        for document in self.documents:
            if prediction := did2prediction.get(document.doc_id):  # (phrase, rel)
                for base_phrase, pred in zip(document.base_phrases, prediction):
                    base_phrase.rels = self._to_rels(pred, document.base_phrases)
            else:
                if skip_untagged is True:
                    continue
                for base_phrase in document.base_phrases:
                    base_phrase.rels = []
            orig_did_to_sentences[to_orig_doc_id(document.doc_id)] += extract_target_sentences(document)

        documents = []
        for sentences in orig_did_to_sentences.values():
            document = Document.from_sentences(sentences)
            documents.append(document)
            if destination is None:
                continue
            output_knp_lines = document.to_knp().splitlines()
            if add_pas_tag is True:
                output_knp_lines = self._add_pas_tag(output_knp_lines, document)
            output_string = "\n".join(output_knp_lines) + "\n"
            if isinstance(destination, Path):
                destination.joinpath(f"{document.doc_id}.knp").write_text(output_string)
            elif isinstance(destination, io.TextIOBase):
                destination.write(output_string)

        return documents

    def _to_rels(
        self,
        prediction: list[int],  # (rel)
        base_phrases: list[BasePhrase],
    ) -> RelTagList:
        rels = RelTagList()
        assert len(self.rel_types) == len(prediction)
        for relation, pred in zip(self.rel_types, prediction):
            if pred < 0:
                continue  # non-target phrase
            if 0 <= pred < len(base_phrases):
                # normal
                prediction_bp: BasePhrase = base_phrases[pred]
                rels.append(
                    RelTag(
                        type=relation,
                        target=prediction_bp.head.text,
                        sid=prediction_bp.sentence.sid,
                        base_phrase_index=prediction_bp.index,
                        mode=None,
                    )
                )
            elif 0 <= pred - len(base_phrases) < len(self.specials):
                # special
                special_arg = self.specials[pred - len(base_phrases)]
                if special_arg in [str(e) for e in self.exophora_referents]:  # exclude [NULL] and [NA]
                    rels.append(
                        RelTag(
                            type=relation,
                            target=special_arg,
                            sid=None,
                            base_phrase_index=None,
                            mode=None,
                        )
                    )
            else:
                raise ValueError(f"invalid pred index: {pred}")

        return rels

    def _add_pas_tag(
        self,
        knp_lines: list[str],
        document: Document,
    ) -> list[str]:
        dtid2pas = {pas.predicate.base_phrase.global_index: pas for pas in document.pas_list()}
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
        sid2index: dict[str, int] = {sent.sid: i for i, sent in enumerate(document.sentences)}
        case_elements = []
        for case in self.cases + ["ノ"] * (CohesionTask.BRIDGING in self.tasks):
            items = ["-"] * 6
            items[0] = case
            args = pas.get_arguments(case, relax=False)
            if args:
                arg: Argument = args[0]
                items[1] = arg.type.value  # フラグ (C/N/O/D/E/U)
                items[2] = str(arg)  # 見出し
                if isinstance(arg, EndophoraArgument):
                    items[3] = str(sid2index[pas.sid] - sid2index[arg.base_phrase.sentence.sid])  # N文前
                    items[4] = str(arg.base_phrase.index)  # tag id
                    items[5] = str(list(arg.base_phrase.entities)[0].eid)  # Entity ID
                else:
                    assert isinstance(arg, ExophoraArgument)
                    items[3] = str(-1)
                    items[4] = str(-1)
                    items[5] = str(arg.eid)  # Entity ID
            else:
                items[1] = "U"
            case_elements.append("/".join(items))
        return f"<述語項構造:{cfid}:{';'.join(case_elements)}>"
