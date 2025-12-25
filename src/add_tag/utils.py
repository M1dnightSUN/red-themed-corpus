from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def setup_logger(name: str = "add_tag", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def chunked(items: Iterable[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    batch: list[T] = []
    chunks: list[list[T]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            chunks.append(batch)
            batch = []
    if batch:
        chunks.append(batch)
    return chunks


def retry(
    func: Callable[[], T],
    *,
    max_retries: int,
    backoff_sec: float,
    jitter_sec: float = 0.3,
    logger: logging.Logger | None = None,
) -> T:
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise
            sleep_for = backoff_sec * (2**attempt) + random.uniform(0, jitter_sec)
            if logger:
                logger.warning(
                    "Retry %s/%s after error: %s; sleep %.2fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    sleep_for,
                )
            time.sleep(sleep_for)
            attempt += 1
