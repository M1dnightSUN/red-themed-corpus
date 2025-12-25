from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import Settings
from tagger import Tagger
from utils import setup_logger


def main() -> None:
    settings = Settings()
    logger = setup_logger()
    input_path = Path("data/test.json")
    output_path = input_path

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {input_path}")

    sample = data[:3]
    tagger = Tagger(settings)
    tagged: list[dict[str, Any]] = []

    for item in tqdm(sample, total=len(sample), desc="Tagging (sample)"):
        if not isinstance(item, dict):
            item = {"value": item}
        item["TAG"] = tagger.tag_item(item)
        tagged.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tagged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s items to %s", len(tagged), output_path)


if __name__ == "__main__":
    main()
