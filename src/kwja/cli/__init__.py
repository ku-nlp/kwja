import os
import sys

if "pytest" not in sys.modules:
    os.environ["KWJA_CLI_MODE"] = "1"
