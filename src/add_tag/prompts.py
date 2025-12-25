from __future__ import annotations

from config import Settings

SYSTEM_PROMPT = (
    "你是一个社会主义核心价值观分类专家。"
    "任务：根据给出的问答对判断该问答对属于社会主义核心价值观的哪些类别。"
    "该问答对包括一个问题、一个正确回答和一个错误回答。"
    "你必须只返回JSON数组，数组元素为允许的类别名称，不能输出其他文本。"
)


def build_user_prompt(
        question: str,
        advocate: str,
        opponent: str,
        settings: Settings | None = None,
) -> str:
    cfg = settings or Settings()
    allowed = "、".join(cfg.allowed_tags)
    return (
        "允许的类别列表如下："
        f"{allowed}\n"
        "请根据以下问答对给出所属类别（可多选），"
        "只返回JSON数组，例如：[\"民主\", \"富强\"]。\n"
        "---问答对如下---\n"
        f"问：{question}\n"
        f"正确回答：{advocate}\n"
        f"错误回答：{opponent}\n"
    )
