import sys, json; from pathlib import Path; results = [];
for r in map(json.load, map(open, Path(".").glob("trace-*.json"))): results += r
json.dump(results, sys.stdout)

