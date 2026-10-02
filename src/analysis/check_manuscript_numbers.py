"""
Check that every decimal value in the manuscript source matches a value generated
by manuscript_numbers.py (outputs/journal_numbers/numbers.json).

Integers, software versions and layout constants are listed in ALLOWED.
Exit status 1 if any value is not found.

Usage:
    python -m src.analysis.check_manuscript_numbers path/to/manuscript.tex [more.tex ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUMBERS = Path(__file__).resolve().parents[2] / "outputs" / "journal_numbers" / "numbers.json"
ALLOWED = {"0.51", "3.11", "0.05", "0.25", "0.70", "0.35", "0.45"}  # versions, alpha, column widths
QUOTED = {"0.8307", "0.9127"}  # conference values quoted as printed in the conference paper


def main(paths: list[str]) -> int:
    values = set()
    for v in json.loads(NUMBERS.read_text()).values():
        values.update(x.lstrip("-") for x in re.findall(r"-?\d+\.\d+", v))
    bad = 0
    for path in paths:
        for i, line in enumerate(Path(path).read_text().splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            for n in re.findall(r"(?<![\d.])-?\d+\.\d{2,4}(?!\d)", line):
                n = n.lstrip("-")
                if n not in values and n not in ALLOWED and n not in QUOTED:
                    print(f"{path}:{i}: {n} not generated")
                    bad += 1
    print(f"{bad} unmatched values")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
