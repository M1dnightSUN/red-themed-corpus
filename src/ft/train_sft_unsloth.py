from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import TrainingArguments

from unsloth import FastLanguageModel
from trl import SFTTrainer

from src.configs.utils_config import load_sft_config


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    返回 output_dir 下最新的 checkpoint-* 路径；若不存在则返回 None。
    """
    p = Path(output_dir)
    if not p.exists():
        return None

    ckpts = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m:
                ckpts.append((int(m.group(1)), str(d)))

    if not ckpts:
        return None

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def _messages_to_text(tokenizer, messages: List[Dict[str, Any]]) -> str:
    """
    将 OpenAI 风格 messages 转为单条训练文本。
    Qwen2.5 Instruct 支持 chat template；这里不添加 generation prompt，
    因为 messages 中已经包含 assistant 的目标输出。
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    # 避免 tokenizer 多进程告警与潜在死锁
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = load_sft_config()

    # -----------------------------
    # 配置字段（按sft.toml 约定）
    # 如字段名与你现有配置不同，在这里改映射即可
    # -----------------------------
    model_path: str = cfg.model_path
    train_jsonl: str = cfg.train_jsonl
    eval_jsonl: str = cfg.eval_jsonl
    output_dir: str = cfg.output_dir

    max_seq_len: int = cfg.max_seq_len
    per_device_batch_size: int = cfg.per_device_batch_size
    grad_accum_steps: int = cfg.grad_accum_steps
    learning_rate: float = cfg.learning_rate
    num_train_epochs: float = cfg.num_train_epochs
    warmup_ratio: float = cfg.warmup_ratio
    lr_scheduler_type: str = cfg.lr_scheduler_type

    logging_steps: int = cfg.logging_steps
    save_steps: int = cfg.save_steps
    eval_steps: int = cfg.eval_steps
    save_total_limit: int = cfg.save_total_limit
    seed: int = cfg.seed

    packing: bool = cfg.packing  # True 会把多条样本 pack 到一个序列，提升吞吐
    lora_r: int = cfg.lora_r
    lora_alpha: int = cfg.lora_alpha
    lora_dropout: float = cfg.lora_dropout

    resume_enabled: bool = cfg.resume_enabled  # True 时自动找 checkpoint 续训

    # -----------------------------
    # dtype / 量化策略
    # -----------------------------
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bool(bf16_supported)
    use_fp16 = not use_bf16

    # -----------------------------
    # 1) 加载模型（本地路径）
    # -----------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        load_in_4bit=True,
    )

    # 添加 LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # -----------------------------
    # 2) 读取数据集（JSONL，每行一个对象，包含 messages）
    # -----------------------------
    data_files = {"train": train_jsonl, "validation": eval_jsonl}
    ds = load_dataset("json", data_files=data_files)

    # 将 messages -> text，供 SFTTrainer 训练
    def map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example["messages"]
        text = _messages_to_text(tokenizer, messages)
        return {"text": text}

    ds = ds.map(
        map_fn,
        remove_columns=[c for c in ds["train"].column_names if c != "messages"],
        desc="Formatting messages with chat template",
    )

    # -----------------------------
    # 3) TrainingArguments（支持断点续训）
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # -----------------------------
    # 4) SFTTrainer
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=packing,
        args=training_args,
    )

    # -----------------------------
    # 5) 自动断点续训（从最新 checkpoint-* 继续）
    # -----------------------------
    resume_ckpt = _find_latest_checkpoint(output_dir) if resume_enabled else None
    if resume_ckpt:
        print(f"[SFT] Resuming from checkpoint: {resume_ckpt}")
    else:
        print("[SFT] Starting training from scratch")

    # -----------------------------
    # 6) 开始训练
    # -----------------------------
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 保存最终 LoRA adapter + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[SFT] Done.")
    print(f"[SFT] Output dir: {output_dir}")


if __name__ == "__main__":
    main()
