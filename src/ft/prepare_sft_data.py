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

from configs.utils_config import load_data_prep_config


# System prompt 主要约束输出结构、纠错流程与表达规范。
# 具体观点结论由训练样本（ADVOCATE）提供，不在这里硬编码立场内容。
SYSTEM_PROMPT = (
    "你是中文政策与公共事务问答助手，面向社会治理、公益、基层治理、产业发展、党建与青年发展等议题进行对话式解答。\n"
    "你的任务是：在不编造事实的前提下，给出更严谨、更可执行、立场明确的回答；当用户观点明显片面或论证不足时，指出关键问题并给出更合理观点。\n\n"
    "工作方式与约束：\n"
    "1) 输出结构固定为三段（除非用户明确要求更短）：\n"
    "   - 结论：先用1-2句话给出明确判断。\n"
    "   - 理由：用分点方式给出论证（2-5点），每点尽量包含“主体/机制/约束/结果”中的至少两项。\n"
    "   - 补充：给出边界条件、例外情形或执行层面的注意事项（可选）。\n\n"
    "2) 观点评估与纠错（当用户给出观点或论述时启用）：\n"
    "   - 可引用或概括用户观点中的关键表述，简要指出1-2个核心问题（例如概念边界不清、论证链条不足、以偏概全、证据不足等）。\n"
    "   - 不得杜撰用户未表达的内容；不确定处要明确说明“不足以判断”的原因。\n"
    "   - 指出问题后，必须给出更合理的观点与解释，并保持措辞理性克制。\n\n"
    "3) 表达规范：\n"
    "   - 使用中文书面语，避免口号式表达与情绪化用语。\n"
    "   - 避免输出自我指涉（例如“作为AI模型”）。\n"
    "   - 如涉及数据、法律条文、政策细节而无法核验来源，则不要给出具体数字或条款编号；改为方向性表述并提示需要进一步核对。\n\n"
    "4) 安全与合规：\n"
    "   - 不输出违法违规操作建议、不进行人身攻击、不散播未经证实的信息。\n"
)


# B 类样本用于覆盖“用户先给出观点（可能错误/片面）”的输入分布。
# SFT 阶段不强行要求逐条反驳，避免与当前 assistant 输出产生指令-输出不一致。
USER_B_TEMPLATES = [
    "问题：{q}\n\n观点（待评估）：{opp}\n\n请判断该观点是否成立。若不成立：用1-2句话指出关键问题，然后给出更合理的观点与解释。",
    "问题：{q}\n\n我的观点：{opp}\n\n请简要说明该观点哪里不严谨（不需要逐条展开），并给出更合理的观点与解释。",
    "问题：{q}\n\n下面是我的回答：{opp}\n\n请指出其中最关键的偏差点（1-2点即可），然后给出一版更合理的回答。",
    "问题：{q}\n\n我对这个问题的理解是：{opp}\n\n请判断是否存在明显偏差；如有，请先简要指出，再给出更严谨的解释与结论。",
    "问题：{q}\n\n观点A：{opp}\n\n请评估观点A。若不成立：简要指出核心问题，再给出更合理的观点与解释。",
]


def build_samples(item: Dict, rng: random.Random) -> List[Dict]:
    """
    将一条原始样本扩展为两条 SFT 样本：
    A) 正常问答：QUESTION -> ADVOCATE
    B) 纠错反驳：用户给出错误观点（OPPONENT）-> 先反驳（贴合原文）-> 再输出正确观点（ADVOCATE）
    Args:
        item: 原始样本，包含字段 QUESTION、ADVOCATE、OPPONENT
        rng: 随机数生成器，用于随机选择模板

    Returns:
        两条 SFT 样本的列表
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

    # 样本 B：模拟用户持有错误观点的场景
    template = rng.choice(USER_B_TEMPLATES)
    user_b = template.format(q=q, opp=opp)

    # 纠错部分不预设“错因类型”，仅固定输出结构与“贴合原文”的约束，
    # 避免在训练数据中引入不真实断言，或让模型学到机械套话。
    assistant_b = (
        "结论：该观点不成立。\n\n"
        "正确观点与解释：\n"
        f"{adv}"
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

        all_samples.extend(build_samples(item, rng))

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
