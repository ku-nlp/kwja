import sys
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rhoknp import Document

sys.setrecursionlimit(10000)


def iter_files(input_paths: list[str], ext: str = "knp") -> Iterator[Path]:
    for path_str in input_paths:
        path = Path(path_str)
        if path.exists() is False:
            continue
        if path.is_dir():
            yield from path.glob(f"**/*.{ext}")
        else:
            yield path


def load_documents(paths: list[Path]) -> list[Document]:
    documents = []
    with ProcessPoolExecutor(8) as executor:
        for document in executor.map(load_document, paths):
            documents.append(document)
    return documents


def load_document(path: Path) -> Document:
    return Document.from_knp(path.read_text())


def count_cases(documents: list[Document]) -> dict[str, int]:
    counter: dict[str, int] = defaultdict(int)
    for document in documents:
        for base_phrase in document.base_phrases:
            for rel_tag in base_phrase.rel_tags:
                counter[rel_tag.type] += 1
    return counter


def main():
    documents = load_documents(sorted(iter_files(sys.argv[1:])))
    counter = count_cases(documents)
    for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print(k, v)


if __name__ == "__main__":
    main()
