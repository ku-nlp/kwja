import sys
from pathlib import Path

from rhoknp import KNP, Document, Sentence

from kwja.metrics.word import WordModuleMetric

CORPORA: tuple[str, ...] = ("kwdlc", "kyoto", "fuman", "wac")


def evaluate_docs(documents: list[Document]):
    word_module_metric = WordModuleMetric(-1)
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
        parsed_sentences: list[Sentence] = []
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
        WordModuleMetric.refresh(partly_gold_document1, level=1)
        try:
            partly_gold_document1 = knp.apply_to_document(partly_gold_document1)
        except ValueError:
            continue

        partly_gold_document2 = document.reparse()
        partly_gold_document2.did = document.did
        WordModuleMetric.refresh(partly_gold_document2, level=2)
        partly_gold_document2 = knp.apply_to_document(partly_gold_document2)

        gold_documents.append(document)
        partly_gold_documents1.append(partly_gold_document1)
        partly_gold_documents2.append(partly_gold_document2)
        predicted_documents.append(predicted_document)

    metrics: dict[str, float] = {}
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


def main():
    for corpus in CORPORA:
        knp_dir = Path(sys.argv[1]) / corpus / "test"
        metrics, num_docs = evaluate_docs(
            [Document.from_knp(knp_path.read_text()) for knp_path in knp_dir.glob("*.knp")]
        )
        print(corpus, num_docs, metrics)


if __name__ == "__main__":
    main()
