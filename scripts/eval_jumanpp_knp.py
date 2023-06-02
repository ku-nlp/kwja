import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rhoknp import KNP, Document, Sentence
from rhoknp.props import MemoTag

from kwja.metrics.word import WordModuleMetric

CORPORA: Tuple[str, ...] = ("kwdlc", "kyoto", "fuman", "wac")


def evaluate_docs(documents: List[Document]):
    word_module_metric = WordModuleMetric()
    knp = KNP(options=["-tab", "-dpnd"])
    (
        gold_documents,
        partly_gold_documents1,
        partly_gold_documents2,
        predicted_documents,
    ) = (
        [],
        [],
        [],
        [],
    )
    for document in documents:
        parsed_sentences: List[Sentence] = []
        for sentence in document.sentences:
            parsed_sentence = knp.apply_to_sentence(sentence.text)
            parsed_sentence.sid = sentence.sid
            parsed_sentence.did = sentence.did
            parsed_sentences.append(parsed_sentence)
        try:
            predicted_document = Document.from_sentences(parsed_sentences)
        except AttributeError:
            continue

        partly_gold_document1 = document.reparse()
        partly_gold_document1.did = document.did
        refresh(partly_gold_document1, level=1)
        try:
            partly_gold_document1 = knp.apply_to_document(partly_gold_document1)
        except ValueError:
            continue

        partly_gold_document2 = document.reparse()
        partly_gold_document2.did = document.did
        refresh(partly_gold_document2, level=2)
        partly_gold_document2 = knp.apply_to_document(partly_gold_document2)

        gold_documents.append(document)
        partly_gold_documents1.append(partly_gold_document1)
        partly_gold_documents2.append(partly_gold_document2)
        predicted_documents.append(predicted_document)

    metrics: Dict[str, float] = {}
    metrics.update(word_module_metric.compute_reading_prediction_metrics(predicted_documents, gold_documents))
    # metrics.update(word_module_metric.compute_morphological_analysis_metrics(predicted_documents, gold_documents))
    # metrics.update(word_module_metric.compute_word_feature_tagging_metrics(predicted_documents, gold_documents))
    metrics.update(word_module_metric.compute_ner_metrics(partly_gold_documents1, gold_documents))
    metrics.update(
        word_module_metric.compute_base_phrase_feature_tagging_metrics(partly_gold_documents1, gold_documents)
    )
    metrics.update(word_module_metric.compute_dependency_parsing_metrics(partly_gold_documents1, gold_documents))
    # metrics.update(word_module_metric.compute_cohesion_analysis_metrics(partly_gold_documents2, gold_documents))
    return metrics, len(gold_documents)


def refresh(document: Document, level: int = 1) -> None:
    """Refresh document

    NOTE:
        level1: clear discourse relations, rel tags, dependencies, and base phrase features.
        level2: clear discourse relations and rel tags.
    """
    assert level in (1, 2), f"invalid level: {level}"
    try:
        for clause in document.clauses:
            clause.discourse_relations.clear()
    except AttributeError:
        pass

    for base_phrase in document.base_phrases:
        base_phrase.rel_tags.clear()
        base_phrase.memo_tag = MemoTag()
        if level == 1:
            base_phrase.features.clear()
            base_phrase.parent_index = None
            base_phrase.dep_type = None

    for phrase in document.phrases:
        if level == 1:
            phrase.features.clear()
            phrase.parent_index = None
            phrase.dep_type = None


def main():
    for corpus in CORPORA:
        knp_dir = Path(sys.argv[1]) / corpus / "test"
        metrics, num_docs = evaluate_docs(
            [Document.from_knp(knp_path.read_text()) for knp_path in knp_dir.glob("*.knp")]
        )
        print(corpus, num_docs, metrics)


if __name__ == "__main__":
    main()
