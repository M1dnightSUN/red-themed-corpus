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

from src.configs.utils_config import load_data_prep_config


# SFT 阶段的 system prompt 尽量与训练监督输出（ADVOCATE）的写作分布一致：
SYSTEM_PROMPT = (
    "你是中文政策与公共事务问答助手。请用中文书面语、语气理性克制地回答问题，并保持论证完整。\n\n"
    "要求：\n"
    "1) 先明确给出立场或结论，再补充解释与论据（可用段落或分点，不强制列点）。\n"
    "2) 不编造事实；不确定时说明依据不足或给出条件化表述。\n"
    "3) 避免口号化与情绪化表达，不要输出“作为AI模型”等自我指涉。\n"
)


# B 类样本用于覆盖“用户先给出错误观点，需要被纠正”的输入分布。
# 由于 OPPONENT 在数据语义上必错，这里将任务描述为“纠错并给出正确观点”，避免条件句带来的歧义。
USER_B_PROMPT = (
    "问题：{q}\n\n"
    "错误观点：{opp}\n\n"
    "请纠正该观点，并给出正确回答。"
)


def build_samples(item: Dict, rng: random.Random, type_b_ratio: float) -> List[Dict]:
    """
    将一条原始样本扩展为两条 SFT 样本：
    A) 正常问答：QUESTION -> ADVOCATE
    B) 反驳样本：QUESTION + OPPONENT（错误观点输入）-> ADVOCATE（按比例生成）
    Args:
        item: 原始样本，包含字段 QUESTION、ADVOCATE、OPPONENT
        rng: 随机数生成器，用于随机选择模板

    Returns:
        两条 SFT 样本的列表
    """
    q = str(item["QUESTION"]).strip()
    adv = str(item["ADVOCATE"]).strip()
    opp = str(item["OPPONENT"]).strip()

    samples: List[Dict] = []

    # 样本 A：直接学习正确回答
    sample_a = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": adv},
        ]
    }

    samples.append(sample_a)

    # 样本B：学习纠正错误观点
    if type_b_ratio > 0 and rng.random() < type_b_ratio:
        user_b = USER_B_PROMPT.format(q=q, opp=opp)

        # 给出最短纠错前缀，主体内容仍为 ADVOCATE
        assistant_b = (
            "该观点不成立。\n\n"
            f"{adv}"
        )

        sample_b = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_b},
                {"role": "assistant", "content": assistant_b},
            ]
        }
        samples.append(sample_b)

    return samples


def main() -> None:
    cfg = load_data_prep_config()

    os.makedirs(cfg.out_dir, exist_ok=True)

    # 读取原始数据（JSON 数组）。训练侧使用 JSONL（逐行 JSON）便于切分与复用。
    with open(cfg.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("input_json 必须是 JSON 数组（list）。")

    if cfg.max_items and cfg.max_items > 0:
        raw = raw[: cfg.max_items]

    rng = random.Random(cfg.seed)

    all_samples: List[Dict] = []
    for item in raw:
        # 字段校验，避免缺字段导致训练数据静默污染
        for k in ("QUESTION", "ADVOCATE", "OPPONENT"):
            if k not in item:
                raise KeyError(f"样本缺少字段 {k}: {item}")

        all_samples.extend(build_samples(item, rng, cfg.type_b_ratio))

    rng.shuffle(all_samples)

    n_eval = max(1, int(len(all_samples) * cfg.eval_ratio))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    train_path = os.path.join(cfg.out_dir, "train.jsonl")
    eval_path = os.path.join(cfg.out_dir, "eval.jsonl")

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
