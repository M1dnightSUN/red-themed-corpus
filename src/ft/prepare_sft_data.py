#!/red-themed-corpus/.venv/bin python
# -*- coding: utf-8 -*-
'''
@Project : red-themed-corpus
@File : prepare_sft_data.py
@Author : tinypolaris
@Date : 2026/1/9 11:40
@Software: PyCharm
'''

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

from configs.utils_config import load_sft_config

# System prompt 用于约束模型的输出风格与纠错行为。
SYSTEM_PROMPT = (
    "你是一名中文问答助手。你需要给出立场明确、逻辑严谨、表达克制的回答。\n"
    "规则：\n"
    "1) 先给出结论，再给出条理化论证（分点）。\n"
    "2) 不编造事实；不确定时使用“可能/通常/取决于”，并说明依据不足。\n"
    "3) 当用户表达明显错误观点时：必须先指出错误点（逐点、具体、贴合用户原文），再给出更合理的观点与解释。\n"
    "4) 回答使用中文，避免空泛口号；指出错误点时不得杜撅用户未表达的内容。"
)


def build_samples(item: Dict) -> List[Dict]:
    """
    将一条原始样本扩展为两条 SFT 训练样本：
    A) 正常问答：QUESTION -> ADVOCATE
    B) 纠错反驳：用户给出错误观点（OPPONENT）-> 先反驳（要求贴合原文）-> 再输出正确观点（ADVOCATE）
    """
    q = str(item["QUESTION"]).strip()
    adv = str(item["ADVOCATE"]).strip()
    opp = str(item["OPPONENT"]).strip()

    # 样本 A：直接学习正确回答
    sample_a = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": adv},
        ]
    }

    # 样本 B：模拟用户持有错误观点的场景，训练“先反驳后纠正”的对话策略
    user_b = (
        f"问题：{q}\n\n"
        f"我的观点：{opp}\n\n"
        "请你评估我的观点是否成立。如果不成立：\n"
        "1) 必须引用或概括我观点中的具体句子/要点来逐点反驳（不要写空泛套话）；\n"
        "2) 然后给出更合理的观点与解释；\n"
        "3) 最后用一句话总结。"
    )

    # 纠错部分不预设“错因类型”，只固定输出结构与“贴合原文”的约束。
    # 这样可以避免训练数据在纠错点上引入不真实断言或形成套话模式。
    assistant_b = (
        "结论：该观点不成立或不够严谨。\n\n"
        "错误点（需引用或概括上面观点中的具体内容逐点说明）：\n"
        "1) …（引用/概括观点中的具体表述A，并说明为何不成立/不充分）\n"
        "2) …（引用/概括观点中的具体表述B，并说明逻辑问题或证据缺口）\n"
        "3) …（如仍有关键问题再补充；若不足三点，可写“无更多关键错误点”）\n\n"
        "更合理的观点与解释：\n"
        f"{adv}\n\n"
        "总结：请基于可验证的事实与清晰的论证链条形成判断。"
    )

    sample_b = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_b},
            {"role": "assistant", "content": assistant_b},
        ]
    }

    return [sample_a, sample_b]


def main() -> None:
    cfg = load_sft_config()

    os.makedirs(cfg.sft_data_dir, exist_ok=True)

    # 读取原始数据（JSON 数组）。训练侧使用 JSONL（逐行 JSON）更便于切分、流式处理与断点复用。
    with open(cfg.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("input_json 必须是 JSON 数组（list）。")

    if cfg.max_items and cfg.max_items > 0:
        raw = raw[: cfg.max_items]

    random.seed(cfg.seed)

    # 扩展样本：每条原始样本生成两条 SFT 训练样本
    all_samples: List[Dict] = []
    for item in raw:
        # 基础字段校验，避免训练时出现隐性缺失导致的脏数据
        for k in ("QUESTION", "ADVOCATE", "OPPONENT"):
            if k not in item:
                raise KeyError(f"样本缺少字段 {k}: {item}")

        all_samples.extend(build_samples(item))

    # 打乱并划分 train/eval
    random.shuffle(all_samples)
    n_eval = max(1, int(len(all_samples) * cfg.eval_ratio))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    train_path = os.path.join(cfg.sft_data_dir, "train.jsonl")
    eval_path = os.path.join(cfg.sft_data_dir, "eval.jsonl")

    # 写出 JSONL：一行一个 JSON 对象
    with open(train_path, "w", encoding="utf-8") as f:
        for x in train_samples:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for x in eval_samples:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(
        f"Prepared SFT data: total={len(all_samples)} "
        f"train={len(train_samples)} eval={len(eval_samples)}"
    )
    print(f"train: {train_path}")
    print(f"eval : {eval_path}")


if __name__ == "__main__":
    main()
