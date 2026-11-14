"""Read-only parsing of the committed outputs/ files."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"


class OutputsMissing(RuntimeError):
    pass


def _num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def read_results(name):
    path = OUT_DIR / name
    if not path.exists():
        raise OutputsMissing(name)
    res = {}
    for line in path.read_text().splitlines():
        if " = " in line:
            k, v = line.split(" = ", 1)
            res[k.strip()] = _num(v.strip())
    return res


def read_table(name):
    path = OUT_DIR / name
    if not path.exists():
        raise OutputsMissing(name)
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]
