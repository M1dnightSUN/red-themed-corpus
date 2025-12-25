#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   json_combine.py
@Time    :   2025/12/25 11:07:11
@Author  :   MidnightSun
@Desc    :   将两个 JSON 数组合并为一个 JSON 数组
'''


import argparse
import json
from pathlib import Path


def _load_json_array(path: Path) -> list:
    """读取 JSON 数组文件并返回其内容

    Args:
        path (Path): JSON 文件路径

    Raises:
        SystemExit: File not found
        SystemExit: Invalid JSON
        SystemExit: JSON is not an array

    Returns:
        list: JSON 数组内容
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {path}")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine two JSON array files by appending the second array "
            "to the first, and write a single combined array."
        )
    )
    parser.add_argument(
        "first",
        nargs="?",
        default="data/qa_1.json",
        help="Path to the first JSON array file (default: data/qa_1.json)",
    )
    parser.add_argument(
        "second",
        nargs="?",
        default="data/qa_2.json",
        help="Path to the second JSON array file (default: data/qa_2.json)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="data/qa_combined.json",
        help="Output path for the combined JSON array (default: data/qa_combined.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    first_path = Path(args.first)
    second_path = Path(args.second)
    output_path = Path(args.output)

    first_items = _load_json_array(first_path)
    second_items = _load_json_array(second_path)
    combined = first_items + second_items

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(combined)} items to {output_path}")


if __name__ == "__main__":
    main()
