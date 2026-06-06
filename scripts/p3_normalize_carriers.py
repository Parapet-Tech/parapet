#!/usr/bin/env python3
"""Thin CLI: normalize AgentDojo/AgentDyn runs into P3 carrier artifacts.

Logic lives in parapet_data.p3 (tested under parapet-data/tests/p3). parapet-data is
editable-installed in the normal flow; this also supports running in place.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parapet-data"))

from parapet_data.p3.carriers.normalize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
