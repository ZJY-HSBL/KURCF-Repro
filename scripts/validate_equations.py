#!/usr/bin/env python3
"""Validate generated README equation SVGs and referenced equation assets.

Only Python's standard library is needed. Run after MathJax generation to
catch malformed SVGs, missing paths, and accidental MathJax error markup.
"""
from __future__ import annotations
from pathlib import Path
from xml.etree import ElementTree as ET
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "equations"
formulae = json.loads((ASSETS / "formulas.json").read_text(encoding="utf-8"))["formulas"]
expected = {item["name"] + ".svg" for item in formulae}
actual = {file.name for file in ASSETS.glob("*.svg")}

problems: list[str] = []
for name in sorted(expected):
    path = ASSETS / name
    if not path.exists():
        problems.append(f"missing image: {name}")
        continue
    try:
        node = ET.parse(path).getroot()
        if node.tag != "{http://www.w3.org/2000/svg}svg":
            problems.append(f"wrong root element: {name}")
        if not any(e.tag.endswith("}path") for e in node.iter()):
            problems.append(f"empty MathJax graphics: {name}")
        if "data-mjx-error" in path.read_text(encoding="utf-8"):
            problems.append(f"math rendering error: {name}")
    except ET.ParseError as ex:
        problems.append(f"invalid / truncated SVG: {name} ({ex})")

problems.extend([f"unexpected image: {n}" for n in sorted(actual - expected)])
for readme in (ROOT / "README.md", ROOT / "README.zh-CN.md"):
    content = readme.read_text(encoding="utf-8")
    references = set(re.findall(r"assets/equations/([\w-]+\.svg)", content))
    if references != expected:
        problems.append(
            f"{readme.name}: missing references {sorted(expected-references)}; "
            f"unexpected references {sorted(references-expected)}"
        )

if problems:
    print("Equation assets FAILED validation:")
    for problem in problems:
        print("-", problem)
    sys.exit(1)
print(f"Validated {len(expected)} SVG equations, XML structure and both READMEs.")
