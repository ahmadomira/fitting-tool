"""
Bootstrap module that imports every sub‑module under the
`core.io.serialize` namespace so that all reader/writer plug‑ins
register themselves via decorators.

It relies on pkgutil.walk_packages to discover Python files located
in the `serialize/` directory that is a sibling of *this* file.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

# Absolute path to the directory that contains this file → core/io
_pkg_root = Path(__file__).resolve().parent

# Directory that holds writer/reader plug‑ins:  core/io/serialize
_serialize_dir = _pkg_root / "serialize"

# Dotted import prefix for discovered modules
_ns_prefix = f"{__package__}.serialize."  # "core.io.serialize."

# Walk over all Python modules inside that directory and import them
for mod in pkgutil.walk_packages([str(_serialize_dir)], prefix=_ns_prefix):
    importlib.import_module(mod.name)
