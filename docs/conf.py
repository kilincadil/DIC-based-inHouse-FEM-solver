# -*- coding: utf-8 -*-
"""Sphinx configuration for the DIC-based in-house FEM documentation."""

import os
import sys
from datetime import date

# Make the source folders importable, so autodoc can find them if it is ever
# enabled. The scripts are plain modules rather than an installed package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _folder in ("solver", "abaqus", "legacy"):
    sys.path.insert(0, os.path.join(_ROOT, _folder))

project = "DIC-based in-house FEM"
author = "Adil Kilinc"
copyright = "%d, %s" % (date.today().year, author)
release = "0.1.0"

extensions = [
    "myst_parser",          # write pages in Markdown
    "sphinx.ext.mathjax",   # render the maths in the model description
    "sphinx.ext.viewcode",
]

# Pages are Markdown; .rst is still accepted if ever needed.
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "DIC-based in-house FEM"
