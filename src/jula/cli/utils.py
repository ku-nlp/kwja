import logging
import warnings

import transformers.utils.logging as hf_logging


def suppress_debug_info() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    hf_logging.set_verbosity(hf_logging.ERROR)