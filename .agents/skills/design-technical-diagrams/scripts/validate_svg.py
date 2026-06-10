#!/usr/bin/env python3
"""Validate structural and accessibility basics for technical SVG figures."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


SVG_NS = "{http://www.w3.org/2000/svg}"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path, nargs="+")
    args = parser.parse_args()
    failures = 0

    for supplied in args.svg:
        path = supplied.resolve()
        errors: list[str] = []
        warnings: list[str] = []

        if not path.is_file():
            print(f"ERROR {path}: file does not exist")
            failures += 1
            continue

        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            print(f"ERROR {path.name}: invalid XML: {exc}")
            failures += 1
            continue

        if local_name(root.tag) != "svg":
            errors.append("root element is not svg")
        for required in ("width", "height", "viewBox"):
            if not root.get(required):
                errors.append(f"missing root attribute: {required}")
        if root.get("role") != "img":
            warnings.append('root should include role="img"')
        if not root.get("aria-label"):
            warnings.append("root should include a meaningful aria-label")

        children = list(root)
        visible_children = [
            child for child in children if local_name(child.tag) not in {"defs", "style"}
        ]
        if not visible_children or local_name(visible_children[0].tag) != "rect":
            warnings.append("first visible element should be an opaque background rect")
        elif visible_children[0].get("fill") in {None, "none", "transparent"}:
            warnings.append("background rect does not have an opaque fill")

        text_nodes = [node for node in root.iter() if local_name(node.tag) == "text"]
        empty_text = [node for node in text_nodes if not "".join(node.itertext()).strip()]
        if empty_text:
            errors.append(f"{len(empty_text)} empty text element(s)")

        marker_ids = {
            node.get("id")
            for node in root.iter()
            if local_name(node.tag) == "marker" and node.get("id")
        }
        for node in root.iter():
            marker_end = node.get("marker-end", "")
            if marker_end.startswith("url(#") and marker_end.endswith(")"):
                marker_id = marker_end[5:-1]
                if marker_id not in marker_ids:
                    errors.append(f"undefined marker reference: {marker_id}")

        for item in errors:
            print(f"ERROR {path.name}: {item}")
        for item in warnings:
            print(f"WARN  {path.name}: {item}")
        print(f"Checked {path.name}: {len(text_nodes)} text elements")
        failures += bool(errors)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
