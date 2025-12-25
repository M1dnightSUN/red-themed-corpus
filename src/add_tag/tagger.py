#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   tagger.py
@Time    :   2025/12/25 16:16:00
@Author  :   MidnightSun
@Desc    :   标签器模块，用于使用大型语言模型为问答对分配标签。
'''


from __future__ import annotations

import ast
import json
from typing import Any

from config import Settings
from llm_client import LLMClient
from prompts import SYSTEM_PROMPT, build_user_prompt
from utils import retry, setup_logger


def _extract_fields(item: dict[str, Any]) -> tuple[str, str, str]:
    """_summary_

    Args:
        item (dict[str, Any]): _description_

    Returns:
        tuple[str, str, str]: _description_
    """
    question = item.get("QUESTION") or ""
    advocate = item.get("ADVOCATE") or ""
    opponent = item.get("OPPONENT") or ""
    return question, advocate, opponent


def _parse_tags(raw_text: str) -> list[str]:
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    # 尝试将文本解析为 JSON。
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = None

    # 尝试将文本解析为 Python 字面量。
    if data is None: 
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start != -1 and end != -1 and start < end: # 找到可能的列表边界
            snippet = raw_text[start : end + 1] # 提取列表部分
            try:
                data = json.loads(snippet) # json 解析
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(snippet) # Python 解析
                except Exception:
                    data = None

    if not isinstance(data, list): 
        return []

    tags: list[str] = [] # 提取字符串标签
    for item in data: 
        if isinstance(item, str):
            tags.append(item.strip())
    return tags


class Tagger:
    def __init__(
        self,
        settings: Settings | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(self.settings)
        self.allowed = set(self.settings.allowed_tags)
        self.logger = setup_logger()

    def tag_item(self, item: dict[str, Any]) -> list[str]:
        existing = item.get("TAG")
        if isinstance(existing, list):
            self.logger.info("已经存在标签，跳过标注。")
            return [tag for tag in existing if isinstance(tag, str)]

        question, advocate, opponent = _extract_fields(item)
        user_prompt = build_user_prompt(question, advocate, opponent, self.settings)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        def _call() -> str:
            return self.client.chat_completion(messages).content

        content = retry(
            _call,
            max_retries=self.settings.max_retries,
            backoff_sec=self.settings.retry_backoff_sec,
            logger=self.logger,
        )
        tags = _parse_tags(content)
        # Filter to allowed tags, keep order and drop duplicates.
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags:
            if tag in self.allowed and tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result
