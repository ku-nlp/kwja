from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from kwja.cli.config import ModelSize
from kwja.cli.utils import download_checkpoint
from kwja.modules.components.deberta_v2 import DebertaV2Model


def main() -> None:
    for model_name in (
        "ku-nlp/deberta-v2-tiny-japanese-char-wwm",
        "ku-nlp/deberta-v2-tiny-japanese",
        "ku-nlp/deberta-v2-base-japanese",
        "google/mt5-small",
        "retrieva-jp/t5-small-short",
    ):
        AutoTokenizer.from_pretrained(model_name)
        AutoConfig.from_pretrained(model_name)

    # Training-step tests load pretrained encoders in addition to the CLI checkpoints.
    AutoModel.from_pretrained("ku-nlp/deberta-v2-tiny-japanese-char-wwm")
    DebertaV2Model.from_pretrained("ku-nlp/deberta-v2-tiny-japanese")
    AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

    for module in ("typo", "char", "seq2seq", "word"):
        download_checkpoint(module, ModelSize.TINY)


if __name__ == "__main__":
    main()
