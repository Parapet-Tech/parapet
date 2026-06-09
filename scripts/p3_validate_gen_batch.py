#!/usr/bin/env python3
"""Thin CLI: validate a P3 generation batch against the carrier backbone + detectors.

Logic lives in parapet_data.p3.generation.validate (tested under parapet-data/tests/p3
with injected fake detectors/carriers). parapet-data is editable-installed in the normal
flow; this also supports running in place.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parapet-data"))

from parapet_data.p3.generation.validate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
