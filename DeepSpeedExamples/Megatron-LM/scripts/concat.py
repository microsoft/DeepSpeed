#!/usr/bin/env python
import json, sys

from pathlib import Path
r = []
for path in Path(".").glob("trace-*.json"):
    with open(path) as f:
        r += json.load(f)
json.dump(r, sys.stdout)
