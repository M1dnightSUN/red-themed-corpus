from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import Settings
from llm_client import LLMClient
from prompts import SYSTEM_PROMPT, build_user_prompt


def _extract_qa(item: dict[str, Any]) -> tuple[str, str, str]:
    question = item.get("QUESTION") or ""
    advocate = item.get("ADVOCATE") or ""
    opponent = item.get("OPPONENT") or ""
    return question, advocate, opponent


def main() -> None:
    settings = Settings()
    data = json.loads(Path(settings.input_path).read_text(encoding="utf-8"))
    sample = data[:3]
    client = LLMClient(settings)

    for idx, item in enumerate(sample, start=1):
        question, advocate, opponent = _extract_qa(item)
        user_prompt = build_user_prompt(question, advocate, opponent, settings)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat_completion(messages)
        print(f"\n=== Sample {idx} ===")
        print(resp.content)


if __name__ == "__main__":
    main()
