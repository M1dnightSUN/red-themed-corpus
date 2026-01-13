from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams


ALLOWED_TAGS = (
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


def main() -> None:
    input_path = Path("data/qa_tagged.json")
    # input_path = Path("data/qa_tagged_sample.json")
    output_path = Path("data/tag_stats.png")

    # 针对中文显示进行配置
    rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    rcParams["axes.unicode_minus"] = False

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {input_path}")

    counter: Counter[str] = Counter()
    allowed = set(ALLOWED_TAGS)
    for item in data:
        if not isinstance(item, dict):
            continue
        tags = item.get("TAG")
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag in allowed:
                counter[tag] += 1

    if not counter:
        raise SystemExit("No valid TAG values found.")

    labels = list(ALLOWED_TAGS)
    values = [counter.get(tag, 0) for tag in labels]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.title("社会主义核心价值观标签统计")
    plt.xlabel("类别")
    plt.ylabel("数量")
    plt.ticklabel_format(axis="y", style="plain")
    
    # 在柱状图上显示数值
    for bar, value in zip(bars, values, strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout() # 调整布局以防止标签被截断

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
