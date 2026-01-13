from __future__ import annotations

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from src.configs.utils_config import load_sft_config, find_latest_checkpoint


CONFIG_PATH = "./configs/sft.toml"


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    cfg = load_sft_config(CONFIG_PATH)
    dtype = pick_dtype()

    train_jsonl = os.path.join(cfg.sft_data_dir, "train.jsonl")
    eval_jsonl  = os.path.join(cfg.sft_data_dir, "eval.jsonl")
    if not os.path.exists(train_jsonl) or not os.path.exists(eval_jsonl):
        raise FileNotFoundError(
            f"未找到 {train_jsonl} 或 {eval_jsonl}，请先运行 prepare_sft_data.py"
        )

    # 1) Load base model from local path (4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_path,
        max_seq_length=cfg.max_seq_len,
        dtype=dtype,
        load_in_4bit=True,
    )

    # 2) LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # 3) Load dataset (jsonl with messages)
    ds = load_dataset("json", data_files={"train": train_jsonl, "eval": eval_jsonl})

    # 4) messages -> text
    def to_text(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    # 5) Training args
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

        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,

        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),

        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,

        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_len,
        packing=cfg.packing,
        args=args,
    )

    resume_ckpt = None
    if cfg.resume:
        resume_ckpt = find_latest_checkpoint(cfg.output_dir)
        if resume_ckpt:
            print(f"Resume from checkpoint: {resume_ckpt}")
        else:
            print("No checkpoint found. Start from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

    # Save adapter + tokenizer to output_dir
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"SFT done. Saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
