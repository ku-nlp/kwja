import os
import platform
import sys

if "pytest" not in sys.modules:
    os.environ["KWJA_CLI_MODE"] = "1"

if platform.system() == "Windows":
    os.environ["PYTHONUTF8"] = "1"
