#!/red-themed-corpus/.venv/bin python
# -*- coding: utf-8 -*-
'''
@Project : red-themed-corpus
@File : utils_config.py
@Author : tinypolaris
@Date : 2026/1/13 11:11
@Software: PyCharm
'''

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # py3.11+
except Exception:
    tomllib = None

try:
    import tomli  # py3.10
except Exception:
    tomli = None


def project_root() -> Path:
    # .../project/src/ft/utils_config.py -> parents[2] == project root
    return Path(__file__).resolve().parents[2]


def load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"TOML config not found: {path}")

    with path.open("rb") as f:
        if tomllib is not None:
            return tomllib.load(f)
        if tomli is None:
            raise RuntimeError("Python<3.11 needs tomli. Please `uv add tomli`.")
        return tomli.load(f)


def find_latest_checkpoint(output_dir: str | Path) -> Optional[str]:
    out = Path(output_dir)
    if not out.is_dir():
        return None

    pattern = re.compile(r"^checkpoint-(\d+)$")
    best_step = -1
    best_path: Optional[Path] = None

    for p in out.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = p

    return str(best_path) if best_path else None


@dataclass
class SFTConfig:
    model_path: str
    input_json: str
    sft_data_dir: str
    output_dir: str

    eval_ratio: float
    seed: int
    max_items: int

    max_seq_len: int
    per_device_batch_size: int
    grad_accum_steps: int
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    lr_scheduler_type: str

    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    packing: bool

    lora_r: int
    lora_alpha: int
    lora_dropout: float

    resume: bool


@dataclass
class DPOConfig:
    base_model_path: str
    sft_adapter_dir: str
    input_json: str
    output_dir: str

    seed: int
    max_items: int

    max_seq_len: int
    per_device_batch_size: int
    grad_accum_steps: int
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    lr_scheduler_type: str

    logging_steps: int
    save_steps: int
    save_total_limit: int

    beta: float
    resume: bool


def _resolve_path(p: str) -> str:
    """
    允许你在 toml 里写相对路径（相对 project root），也允许写绝对路径。
    """
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(project_root() / path)


def load_sft_config() -> SFTConfig:
    cfg_path = project_root() / "src" / "configs" / "sft.toml"
    cfg = load_toml(cfg_path)

    return SFTConfig(
        model_path=_resolve_path(str(cfg["paths"]["model_path"])),
        input_json=_resolve_path(str(cfg["paths"]["input_json"])),
        sft_data_dir=_resolve_path(str(cfg["paths"]["sft_data_dir"])),
        output_dir=_resolve_path(str(cfg["paths"]["output_dir"])),

        eval_ratio=float(cfg["data"]["eval_ratio"]),
        seed=int(cfg["data"]["seed"]),
        max_items=int(cfg["data"].get("max_items", -1)),

        max_seq_len=int(cfg["train"]["max_seq_len"]),
        per_device_batch_size=int(cfg["train"]["per_device_batch_size"]),
        grad_accum_steps=int(cfg["train"]["grad_accum_steps"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=int(cfg["train"]["num_train_epochs"]),
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        lr_scheduler_type=str(cfg["train"]["lr_scheduler_type"]),

        logging_steps=int(cfg["train"]["logging_steps"]),
        save_steps=int(cfg["train"]["save_steps"]),
        eval_steps=int(cfg["train"]["eval_steps"]),
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        packing=bool(cfg["train"]["packing"]),

        lora_r=int(cfg["train"]["lora_r"]),
        lora_alpha=int(cfg["train"]["lora_alpha"]),
        lora_dropout=float(cfg["train"]["lora_dropout"]),

        resume=bool(cfg["train"]["resume"]),
    )


def load_dpo_config() -> DPOConfig:
    cfg_path = project_root() / "src" / "configs" / "dpo.toml"
    cfg = load_toml(cfg_path)

    return DPOConfig(
        base_model_path=_resolve_path(str(cfg["paths"]["base_model_path"])),
        sft_adapter_dir=_resolve_path(str(cfg["paths"]["sft_adapter_dir"])),
        input_json=_resolve_path(str(cfg["paths"]["input_json"])),
        output_dir=_resolve_path(str(cfg["paths"]["output_dir"])),

        seed=int(cfg["data"]["seed"]),
        max_items=int(cfg["data"].get("max_items", -1)),

        max_seq_len=int(cfg["train"]["max_seq_len"]),
        per_device_batch_size=int(cfg["train"]["per_device_batch_size"]),
        grad_accum_steps=int(cfg["train"]["grad_accum_steps"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=int(cfg["train"]["num_train_epochs"]),
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        lr_scheduler_type=str(cfg["train"]["lr_scheduler_type"]),

        logging_steps=int(cfg["train"]["logging_steps"]),
        save_steps=int(cfg["train"]["save_steps"]),
        save_total_limit=int(cfg["train"]["save_total_limit"]),

        beta=float(cfg["train"]["beta"]),
        resume=bool(cfg["train"]["resume"]),
    )
