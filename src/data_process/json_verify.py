#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   json_verify.py
@Time    :   2025/12/25 11:11:28
@Author  :   MidnightSun
@Desc    :   验证合并后的 JSON 数组文件
'''


import json
from pathlib import Path

def verify_length(json_path: Path) -> int:
    """获取json数组长度

    Args:
        json_path (Path): json文件路径

    Raises:
        SystemExit: File not found
        SystemExit: Invalid JSON
        SystemExit: JSON is not an array

    Returns:
        int: Length of the JSON array
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {json_path}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {json_path}")

    return len(data)


if __name__ == "__main__":
    qa_1_path = Path("data/qa_1.json")
    qa_2_path = Path("data/qa_2.json")
    qa_combined_path = Path("data/qa_combined.json")

    print("Verifying qa_1.json: {}".format(qa_1_path))
    len_qa1 = verify_length(qa_1_path) # 27404
    print("{}'s length: {}".format(qa_1_path, len_qa1))

    print("Verifying qa_2.json: {}".format(qa_2_path))
    len_qa2 = verify_length(qa_2_path) # 30001
    print("{}'s length: {}".format(qa_2_path, len_qa2))

    print("Verifying qa_combined.json: {}".format(qa_combined_path))
    len_combined = verify_length(qa_combined_path) # 57405
    print("{}'s length: {}".format(qa_combined_path, len_combined))

    print("length verification: {}.".format(len_qa1 + len_qa2 == len_combined))
