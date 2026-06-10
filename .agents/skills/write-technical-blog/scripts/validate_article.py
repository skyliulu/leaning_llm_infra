#!/usr/bin/env python3
"""Run lightweight structural checks on a Markdown technical article."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
WINDOWS_PATH_RE = re.compile(r"(?i)(?:[a-z]:\\|file://)")
TRANSCRIPT_PHRASES = re.compile(r"(课件中|课程中|本书中|讲义中)(?:提到|指出|说|认为)?")


def is_remote(target: str) -> bool:
    return target.startswith(("http://", "https://", "data:"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("article", type=Path)
    args = parser.parse_args()

    article = args.article.resolve()
    if not article.is_file():
        print(f"ERROR article does not exist: {article}")
        return 2

    text = article.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    if not re.search(r"(?m)^#\s+\S", text):
        errors.append("missing H1 title")
    if "## 参考" not in text and "## References" not in text:
        warnings.append("no references section detected")
    if WINDOWS_PATH_RE.search(text):
        errors.append("article contains a local Windows/file URL")

    for match in IMAGE_RE.finditer(text):
        target = match.group(1).strip().split("#", 1)[0]
        if is_remote(target):
            continue
        image = (article.parent / target).resolve()
        if not image.is_file():
            errors.append(f"missing image: {target}")

    phrases = sorted(set(TRANSCRIPT_PHRASES.findall(text)))
    if phrases:
        warnings.append("source-transcript phrasing detected; state knowledge directly")

    math_open = re.findall(r"(?m)^\s*\\\[\s*$", text)
    math_close = re.findall(r"(?m)^\s*\\\]\s*$", text)
    math_blocks = len(math_open)
    if len(math_close) != math_blocks:
        errors.append("unbalanced display-math delimiters")

    headings = re.findall(r"(?m)^(#{2,6})\s+(.+)$", text)
    for index in range(1, len(headings)):
        previous_level = len(headings[index - 1][0])
        current_level = len(headings[index][0])
        if current_level > previous_level + 1:
            warnings.append(
                f"heading level jumps before: {headings[index][1].strip()}"
            )

    for item in errors:
        print(f"ERROR {item}")
    for item in warnings:
        print(f"WARN  {item}")

    image_count = len(IMAGE_RE.findall(text))
    print(
        f"Checked {article.name}: {len(text.splitlines())} lines, "
        f"{image_count} images, {math_blocks} display-math blocks"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
