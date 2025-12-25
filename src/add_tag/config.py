#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2025/12/25 14:54:48
@Author  :   MidnightSun
@Desc    :   配置文件
'''


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # 百炼API
    # base_url: str = "DASHSCOPE_BASE_URL"
    # model_name: str = "DASHSCOPE_MODEL_NAME"
    # api_key: str = "DASHSCOPE_API_KEY"

    # DeepSeek API
    base_url: str = "DEEPSEEK_BASE_URL"
    model_name: str = "DEEPSEEK_MODEL_NAME"
    api_key: str = "DEEPSEEK_API_KEY"

    # 请求设置
    max_concurrency: int = 50 # 最大并发请求数
    target_qps: int = 50 # 目标每秒请求数
    request_timeout_sec: int = 1800 # 请求超时时间，单位秒

    # 重试设置
    max_retries: int = 5
    retry_backoff_sec: float = 1.0

    # 数据路径
    # input_path: Path = Path("data/qa_combined.json")
    input_path: Path = Path("data/qa_combined.json")
    output_path: Path = Path("data/qa_tagged.json")

    # 允许的标签
    allowed_tags: tuple[str, ...] = (
        "富强",
        "民主",
        "文明",
        "和谐",
        "自由",
        "平等",
        "公正",
        "法治",
        "爱国",
        "敬业",
        "诚信",
        "友善",
    )


# def get_api_key(env_name: str | None = None) -> str:
#     name = env_name or Settings().api_key_env
#     value = os.getenv(name)
#     if not value:
#         raise RuntimeError(f"Missing API key. Set environment variable: {name}")
#     return value

def get_env(env_name: str) -> str:
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {env_name}")
    return value
