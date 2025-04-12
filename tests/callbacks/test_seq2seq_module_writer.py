import tempfile
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Optional, Union

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from kwja.callbacks import Seq2SeqModuleWriter
from kwja.datamodule.datasets import Seq2SeqInferenceDataset
from kwja.utils.constants import CANON_TOKEN, LEMMA_TOKEN, READING_TOKEN, SURF_TOKEN


class MockTrainer:
    def __init__(self, predict_dataloaders: list[DataLoader]) -> None:
        self.predict_dataloaders = predict_dataloaders


@pytest.mark.parametrize(
    "destination",
    [
        None,
        Path(TemporaryDirectory().name) / Path("seq2seq_prediction.seq2seq"),
        str(Path(TemporaryDirectory().name) / Path("seq2seq_prediction.seq2seq")),
    ],
)
def test_init(destination: Optional[Union[str, Path]], seq2seq_tokenizer: PreTrainedTokenizerFast) -> None:
    _ = Seq2SeqModuleWriter(tokenizer=seq2seq_tokenizer, destination=destination)


def test_write_on_batch_end(seq2seq_tokenizer: PreTrainedTokenizerFast) -> None:
    max_src_length = 32
    max_tgt_length = 128
    doc_id_prefix = "test"
    juman_text = dedent(
        """\
        # S-ID:test-0-1
        太郎 _ 太郎 未定義語 15 その他 1 * 0 * 0
        と _ と 未定義語 15 その他 1 * 0 * 0
        次郎 _ 次郎 未定義語 15 その他 1 * 0 * 0
        は _ は 未定義語 15 その他 1 * 0 * 0
        よく _ よく 未定義語 15 その他 1 * 0 * 0
        けんか _ けんか 未定義語 15 その他 1 * 0 * 0
        する _ する 未定義語 15 その他 1 * 0 * 0
        EOS
        # S-ID:test-1-1
        辛い _ 辛い 未定義語 15 その他 1 * 0 * 0
        ラーメン _ ラーメン 未定義語 15 その他 1 * 0 * 0
        が _ が 未定義語 15 その他 1 * 0 * 0
        好きな _ 好きな 未定義語 15 その他 1 * 0 * 0
        ので _ ので 未定義語 15 その他 1 * 0 * 0
        頼み _ 頼み 未定義語 15 その他 1 * 0 * 0
        ました _ ました 未定義語 15 その他 1 * 0 * 0
        EOS
        """
    )
    juman_file = tempfile.NamedTemporaryFile("wt")
    juman_file.write(juman_text)
    juman_file.seek(0)

    dataset = Seq2SeqInferenceDataset(
        seq2seq_tokenizer,
        max_src_length,
        max_tgt_length,
        juman_file=Path(juman_file.name),
    )
    num_examples = len(dataset)

    trainer = MockTrainer([DataLoader(dataset, batch_size=num_examples)])

    generated_texts = [
        f"{SURF_TOKEN}太郎{READING_TOKEN}たろう{LEMMA_TOKEN}太郎{CANON_TOKEN}太郎/たろう"
        f"{SURF_TOKEN}と{READING_TOKEN}と{LEMMA_TOKEN}と{CANON_TOKEN}と/と"
        f"{SURF_TOKEN}次郎{READING_TOKEN}じろう{LEMMA_TOKEN}次郎{CANON_TOKEN}次郎/じろう"
        f"{SURF_TOKEN}は{READING_TOKEN}は{LEMMA_TOKEN}は{CANON_TOKEN}は/は"
        f"{SURF_TOKEN}よく{READING_TOKEN}よく{LEMMA_TOKEN}よい{CANON_TOKEN}良い/よい"
        f"{SURF_TOKEN}けんか{READING_TOKEN}けんか{LEMMA_TOKEN}けんか{CANON_TOKEN}喧嘩/けんか"
        f"{SURF_TOKEN}する{READING_TOKEN}する{LEMMA_TOKEN}する{CANON_TOKEN}する/する",
        f"{SURF_TOKEN}辛い{READING_TOKEN}からい{LEMMA_TOKEN}辛い{CANON_TOKEN}辛い/からい"
        f"{SURF_TOKEN}ラーメン{READING_TOKEN}らーめん{LEMMA_TOKEN}ラーメン{CANON_TOKEN}ラーメン/らーめん"
        f"{SURF_TOKEN}が{READING_TOKEN}が{LEMMA_TOKEN}が{CANON_TOKEN}が/が"
        f"{SURF_TOKEN}好きな{READING_TOKEN}すきな{LEMMA_TOKEN}好きだ{CANON_TOKEN}好きだ/すきだ"
        f"{SURF_TOKEN}ので{READING_TOKEN}ので{LEMMA_TOKEN}のだ{CANON_TOKEN}のだ/のだ"
        f"{SURF_TOKEN}頼み{READING_TOKEN}たのみ{LEMMA_TOKEN}頼む{CANON_TOKEN}頼む/たのむ"
        f"{SURF_TOKEN}ました{READING_TOKEN}ました{LEMMA_TOKEN}ます{CANON_TOKEN}ます/ます",
    ]
    seq2seq_predictions = torch.zeros((num_examples, max_tgt_length), dtype=torch.long)
    for i, generated_text in enumerate(generated_texts):
        for j, output_id in enumerate(seq2seq_tokenizer.encode(generated_text)):
            seq2seq_predictions[i, j] = output_id

    prediction = {
        "example_ids": torch.arange(num_examples, dtype=torch.long),
        "seq2seq_predictions": seq2seq_predictions,
    }

    with TemporaryDirectory() as tmp_dir:
        writer = Seq2SeqModuleWriter(
            tokenizer=seq2seq_tokenizer,
            destination=tmp_dir / Path("seq2seq_prediction.seq2seq"),
        )
        writer.write_on_batch_end(trainer, ..., prediction, None, ..., 0, 0)  # type: ignore
        assert isinstance(writer.destination, Path), "destination isn't set"
        assert writer.destination.read_text() == dedent(
            f"""\
            # S-ID:{doc_id_prefix}-0-1 kwja:{version("kwja")}
            太郎 たろう 太郎 未定義語 15 その他 1 * 0 * 0 "代表表記:太郎/たろう"
            と と と 未定義語 15 その他 1 * 0 * 0 "代表表記:と/と"
            次郎 じろう 次郎 未定義語 15 その他 1 * 0 * 0 "代表表記:次郎/じろう"
            は は は 未定義語 15 その他 1 * 0 * 0 "代表表記:は/は"
            よく よく よい 未定義語 15 その他 1 * 0 * 0 "代表表記:良い/よい"
            けんか けんか けんか 未定義語 15 その他 1 * 0 * 0 "代表表記:喧嘩/けんか"
            する する する 未定義語 15 その他 1 * 0 * 0 "代表表記:する/する"
            EOS
            # S-ID:{doc_id_prefix}-1-1 kwja:{version("kwja")}
            辛い からい 辛い 未定義語 15 その他 1 * 0 * 0 "代表表記:辛い/からい"
            ラーメン らーめん ラーメン 未定義語 15 その他 1 * 0 * 0 "代表表記:ラーメン/らーめん"
            が が が 未定義語 15 その他 1 * 0 * 0 "代表表記:が/が"
            好きな すきな 好きだ 未定義語 15 その他 1 * 0 * 0 "代表表記:好きだ/すきだ"
            ので ので のだ 未定義語 15 その他 1 * 0 * 0 "代表表記:のだ/のだ"
            頼み たのみ 頼む 未定義語 15 その他 1 * 0 * 0 "代表表記:頼む/たのむ"
            ました ました ます 未定義語 15 その他 1 * 0 * 0 "代表表記:ます/ます"
            EOS
            """
        )
