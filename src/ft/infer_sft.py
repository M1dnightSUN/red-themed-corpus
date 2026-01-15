#!/red-themed-corpus/.venv/bin python
# -*- coding: utf-8 -*-
'''
@Project : red-themed-corpus
@File : infer_sft.py
@Author : tinypolaris
@Date : 2026/1/15 09:14
@Software: PyCharm
'''

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


SYSTEM_PROMPT = (
    "你是中文公共事务问答助手。\n"
    "用中文书面语直接回答问题，重点输出观点与解释。\n"
    "不编造事实；不确定时说明依据不足；避免口号化与自我指涉。\n"
)


def chat(tokenizer, model, user_text: str, max_new_tokens=512, temperature=0.2, top_p=0.9) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    base_model_path = r"D:/models/Qwen2.5-7B-Instruct"
    adapter_path = r"data/ckpt_sft"  # 你的训练输出目录（LoRA adapter）

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    # 测试 A：正常提问
    q1 = "你认为中国的公益领域中，谁是最主要的问责对象？为什么？"
    print("\n===== A: QUESTION =====\n")
    print(chat(tokenizer, model, q1))

    # 测试 B：直接给“错误回答/错误观点”，要求纠错（更贴近真实使用）
    opp = (
        "中国的公益领域中，最主要的问责对象应该是政府和企业。政府在公益领域中拥有巨大的权力和资源，"
        "应该承担更多的责任。然而，政府在问责方面往往表现得比较困难，缺乏透明度和有效性。企业作为公益活动的赞助者和参与者，"
        "同样应该承担问责责任……（此处可粘贴完整 OPPONENT）"
    )
    q2 = opp + "\n\n请指出上述观点的问题，并给出正确结论与解释。"
    print("\n===== B: WRONG VIEWPOINT =====\n")
    print(chat(tokenizer, model, q2))


if __name__ == "__main__":
    main()
