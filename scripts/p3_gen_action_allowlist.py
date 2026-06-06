#!/usr/bin/env python3
"""Thin CLI: generate/validate the P3 action-tool allowlist.

Logic lives in parapet_data.p3 (tested under parapet-data/tests/p3). parapet-data is
editable-installed in the normal flow; this also supports running in place.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parapet-data"))

from parapet_data.p3.carriers.allowlist_generator import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
