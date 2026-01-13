#!/red-themed-corpus/.venv/bin python
# -*- coding: utf-8 -*-
'''
@Project : red-themed-corpus
@File : prepare_sft_data.py
@Author : tinypolaris
@Date : 2026/1/9 11:40
@Software: PyCharm
'''

import os
import json
import random
from typing import List, Dict

def build_sft_samples(item: Dict) -> List[Dict]:
    """
    每条数据扩展为 2 条 SFT 样本：
    A: 正常问答（QUESTION -> ADVOCATE）
    B: 反驳错误观点（OPPONENT -> 反驳 + ADVOCATE）
    Args:
        item: 单条数据，包含 "QUESTION", "ADVOCATE", "OPPONENT", 可选 "TAG"

    Returns: 包含两条 SFT 样本的列表

    """
    q = item["QUESTION"].strip()
    adv = item["ADVOCATE"].strip()
    opp = item["OPPONENT"].strip()
    tags = item.get("TAG", [])
    tag_str = "、".join(tags) if tags else "（未标注）"

    system = (
        "你是一名对社会主义核心价值观方面问答的专家和教授。你需要给出立场明确、逻辑严谨、表达克制的回答。\n"
        "规则：\n"
        "1) 先给出结论，再给出条理化论证（分点）。\n"
        "2) 不编造事实；不确定时用“可能/通常/取决于”并说明依据不足。\n"
        "3) 当用户表达明显错误观点时：先指出错误点（分点、具体），再给出更合理的观点与解释。\n"
        "4) 回答使用中文，不输出无意义口号，尽量给出可核查的推理链条（但不要暴露隐私或无来源细节）。\n"
    )

    # 样本 A：正常问答
    sample_a = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"相关价值观：{tag_str}\n问题：{q}"},
            {"role": "assistant", "content": adv},
        ]
    }

    # 样本 B：反驳错误观点
    # 这里用模板让模型学会“先纠错，再给正确观点”
    rebut_answer = (
        "你的观点存在不准确或片面之处，主要问题包括：\n"
        "1) 论据与结论之间缺少充分支撑；\n"
        "2) 对关键主体、责任边界或机制的理解存在偏差；\n"
        "3) 忽略了现实治理与制度安排中的约束条件。\n\n"
        "更合理的观点与解释如下：\n"
        f"{adv}"
    )

    sample_b = {
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"相关价值观：{tag_str}\n"
                    f"问题：{q}\n"
                    f"我的观点：{opp}\n\n"
                    "请你判断我的观点是否正确。如果不正确：\n"
                    "（1）指出错误点；（2）给出更合理的观点与解释。"
                ),
            },
            {"role": "assistant", "content": rebut_answer},
        ]
    }

    return [sample_a, sample_b]


def main(
    input_json: str,
    out_dir: str = "./data_sft",
    eval_ratio: float = 0.02, # 2% 作为评估集
    seed: int = 42,
    max_items: int = -1, # 调试用，>0 则只取前 N 条
):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入必须是 JSON 数组。")

    if max_items > 0:
        data = data[:max_items]

    all_samples = []
    for item in data:
        all_samples.extend(build_sft_samples(item))

    random.shuffle(all_samples)

    n_eval = max(1, int(len(all_samples) * eval_ratio))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    train_path = os.path.join(out_dir, "train.jsonl")
    eval_path = os.path.join(out_dir, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for x in train_samples:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for x in eval_samples:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Total samples: {len(all_samples)} (train={len(train_samples)}, eval={len(eval_samples)})")
    print(f"Saved: {train_path}")
    print(f"Saved: {eval_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="原始红色语料 JSON 数组文件路径")
    parser.add_argument("--out_dir", type=str, default="./data_sft")
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=-1, help="调试用，>0 则只取前 N 条")
    args = parser.parse_args()

    main(
        input_json=args.input_json,
        out_dir=args.out_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_items=args.max_items,
    )
