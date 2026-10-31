"""pytest setup: the tests import the xkan package of this checkout (the folder above tests/); with
XKAN_TESTS_INSTALLED=1 they import the installed one instead (e.g. to test a built wheel)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
if os.environ.get("XKAN_TESTS_INSTALLED") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
