from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import Settings
from tagger import Tagger
from utils import setup_logger


def _atomic_write_json(path: Path, data: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def main() -> None:
    settings = Settings()
    logger = setup_logger()
    input_path = Path(settings.input_path)
    output_path = input_path

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {input_path}")

    tagger = Tagger(settings)
    tagged: list[dict[str, Any]] = []
    total = len(data)

    for idx, item in enumerate(tqdm(data, total=total, desc="Tagging"), start=1):
        if not isinstance(item, dict):
            item = {"value": item}
        item["TAG"] = tagger.tag_item(item)
        tagged.append(item)
        if idx % 10 == 0 or idx == total:
            _atomic_write_json(output_path, tagged)
            logger.info("Tagged %s/%s", idx, total)

    logger.info("Wrote %s items to %s", len(tagged), output_path)


if __name__ == "__main__":
    main()
