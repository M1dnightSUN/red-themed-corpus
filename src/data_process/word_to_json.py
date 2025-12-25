#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   word_to_json.py
@Time    :   2025/12/24 17:24:33
@Author  :   MidnightSun 
@Desc    :   将包含原始 JSON 文本的 .docx 文件转换为 .json 文件
'''


import argparse
import os

from docx import Document


def extract_docx_text(docx_path: str) -> str:
    """提取 docx 的文本内容

    Args:
        docx_path (str): .docx 文件路径

    Raises:
        ValueError: 当文件不是 .docx 格式时抛出

    Returns:
        str: 提取的文本内容
    """
    if not docx_path.lower().endswith(".docx"):
        raise ValueError("Only .docx files are supported.")

    document = Document(docx_path) # 打开 .docx 文件
    paragraphs = [para.text for para in document.paragraphs] # 提取所有段落文本
    return "\n".join(paragraphs) # 合并为单一字符串


def convert_word_to_json(docx_path: str, output_path: str) -> None:
    """将包含原始 JSON 文本的 .docx 文件内容写入 .json 文件

    Args:
        docx_path (str): .docx 文件路径
        output_path (str): 输出的 .json 文件路径
    """
    raw_text = extract_docx_text(docx_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw_text)


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        argparse.Namespace: 包含解析后的命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(
        description="Convert a .docx file containing raw JSON text into a .json file."
    )
    parser.add_argument("input", help="Path to the .docx file containing JSON text")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .json path (default: same basename as input)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args() 
    output_path = args.output 
    # 如果没有指定输出路径，则使用输入文件的基本名称
    if not output_path:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}.json"

    convert_word_to_json(args.input, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
