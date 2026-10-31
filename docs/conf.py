"""Sphinx configuration of xKAN's documentation: the pages, the API from the docstrings, and the tutorials rendered
from their saved outputs (copied in from ../tutorials at each build; nothing is executed).

    pip install -e ".[docs]"; cd docs; make html
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import xkan  # noqa: E402

project = "xKAN"
author = "xKAN contributors (pykan: Ziming Liu)"
copyright = "2024, Ziming Liu; 2026, xKAN contributors"
release = xkan.__version__

extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.viewcode"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_title = f"xKAN {release}"

# the notebooks are shown with the outputs saved in them
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
suppress_warnings = ["myst.header", "mystnb.unknown_mime_type", "misc.highlighting_failure"]

# the tutorials and hellokan, copied into _notebooks/ (kept out of the repository)
notebooks = HERE / "_notebooks"
shutil.rmtree(notebooks, ignore_errors=True)
shutil.copytree(ROOT / "tutorials", notebooks, ignore=shutil.ignore_patterns(".ipynb_checkpoints", "*.py", "*.csv",
                                                                                 "figures", "__pycache__", "runs"))
shutil.copy(ROOT / "hellokan.ipynb", notebooks / "hellokan.ipynb")


def _preformatted(app, what, name, obj, options, lines):
    # the docstrings are plain text (aligned lists and formulas), shown as written rather than parsed as reST
    if lines:
        lines[:] = ["::", ""] + ["    " + line for line in lines] + [""]


def setup(app):
    app.connect("autodoc-process-docstring", _preformatted)
