"""Put the parapet repo root on sys.path so `import p3detectors` / `import p3carriers`
resolve regardless of cwd or install state (no editable install in this repo)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
