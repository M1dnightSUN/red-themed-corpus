#!/red-themed-corpus/.venv/bin python
# -*- coding: utf-8 -*-
'''
@Project : red-themed-corpus
@File : train_dpo_unsloth.py.py
@Author : tinypolaris
@Date : 2026/1/13 11:12
@Software: PyCharm
'''

from __future__ import annotations

import os
import json
import random
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel
from peft import PeftModel

from src.configs.utils_config import load_dpo_config, find_latest_checkpoint


CONFIG_PATH = "./configs/dpo.toml"


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_dpo_dataset(input_json: str, seed: int, max_items: int) -> Dataset:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input_json 必须是 JSON 数组（list）。")

    if max_items and max_items > 0:
        data = data[:max_items]

    random.seed(seed)
    random.shuffle(data)

    system = (
        "你是中文问答助手。你必须给出更合理、更符合事实逻辑的观点与解释。"
        "避免输出明显片面、缺乏依据或与常识/制度安排相冲突的答案。"
    )

    rows = []
    for item in data:
        q = item["QUESTION"].strip()
        chosen = item["ADVOCATE"].strip()
        rejected = item["OPPONENT"].strip()
        rows.append(
            {
                "prompt_messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": q},
                ],
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return Dataset.from_list(rows)


def main():
    cfg = load_dpo_config(CONFIG_PATH)
    dtype = pick_dtype()

    # 1) Load base model (4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model_path,
        max_seq_length=cfg.max_seq_len,
        dtype=dtype,
        load_in_4bit=True,
    )

    # 2) Load SFT adapter as trainable
    # 这里用 PeftModel.from_pretrained 更稳，避免不同版本的 load_adapter API 差异
    model = PeftModel.from_pretrained(
        model,
        cfg.sft_adapter_dir,
        is_trainable=True,
    )

    # 3) Build dataset
    ds = build_dpo_dataset(cfg.input_json, seed=cfg.seed, max_items=cfg.max_items)

    def format_prompt(ex):
        prompt = tokenizer.apply_chat_template(
            ex["prompt_messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt}

    ds = ds.map(format_prompt)

    # 4) Training args
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,

        logging_steps=cfg.logging_steps,

        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),

        optim="adamw_torch",
        report_to="none",
        seed=cfg.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,

        beta=cfg.beta,

        prompt_field="prompt",
        chosen_field="chosen",
        rejected_field="rejected",

        max_length=cfg.max_seq_len,
        max_prompt_length=min(1024, cfg.max_seq_len // 2),
    )

    resume_ckpt = None
    if cfg.resume:
        resume_ckpt = find_latest_checkpoint(cfg.output_dir)
        if resume_ckpt:
            print(f"Resume from checkpoint: {resume_ckpt}")
        else:
            print("No checkpoint found. Start from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"DPO done. Saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
