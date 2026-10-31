"""Execute the tutorials and report the ones that fail (needs nbclient and ipykernel).

python tutorials/run_tutorials.py [notebook ...] [--timeout SECONDS] [--write]

Without notebooks, every notebook under tutorials/. Each runs in its own folder (relative paths as in Jupyter);
--write saves the executed notebooks with their outputs.
"""
import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent


def run(path, timeout, write):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    t0 = time.perf_counter()
    try:
        client.execute()
        error = None
    except CellExecutionError as e:
        error = str(e).strip().splitlines()[-1][:300]
    except Exception as e:  # a timeout or a dead kernel
        error = f"{type(e).__name__}: {str(e)[:300]}"
    if write:
        nbformat.write(nb, path)
    return error, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    paths = [Path(p).resolve() for p in args.notebooks] or sorted(HERE.rglob("*.ipynb"))
    failed = 0
    for path in paths:
        if ".ipynb_checkpoints" in path.parts:
            continue
        error, seconds = run(path, args.timeout, args.write)
        failed += error is not None
        print(f"{'FAIL' if error else 'ok  '} {seconds:7.1f}s {path.relative_to(HERE.parent) if HERE.parent in path.parents else path}"
              + (f"\n      {error}" if error else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
