#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   show_samples.py
@Time    :   2025/12/25 11:22:16
@Author  :   MidnightSun
@Desc    :   显示QA JSON数组文件中的随机元素
'''


import argparse
import json
import random
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a random element from a QA JSON array file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data/qa_combined.json",
        help="Path to the QA JSON file (default: data/qa_combined.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    sample = random.choice(data)
    print(json.dumps(sample, ensure_ascii=False, indent=2))


def show_random_sample(json_path: Path) -> None:
    """显示 JSON 数组文件中的随机元素

    Args:
        json_path (Path): JSON 文件路径
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sample = random.choice(data)
    print(json.dumps(sample, ensure_ascii=False, indent=2))


def iter_samples(json_path: Path):
    """迭代 JSON 数组文件中的所有元素

    Args:
        json_path (Path): JSON 文件路径

    Yields:
        dict: JSON 数组中的每个元素
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for item in data:
        yield item




if __name__ == "__main__":

    path = Path("data/qa_combined.json")
    print("\nShowing a random sample from {}:".format(path))
    show_random_sample(path)

    print("*"*50)
    
    print("\nIterating over all samples in {}:".format(path))
    for i, sample in enumerate(iter_samples(path), 1):
        print(f"Sample {i}: {json.dumps(sample, ensure_ascii=False)}")
        if i >= 5:  # 仅显示前5个样本作为示例
            break