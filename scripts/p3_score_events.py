#!/usr/bin/env python3
"""Thin CLI: score per-event tool-call args from staged carriers (D_gen MLX judge).

Logic lives in parapet_data.p3 (tested under parapet-data/tests/p3). parapet-data is
editable-installed in the normal flow; this also supports running in place.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parapet-data"))

from parapet_data.p3.detectors.score_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
