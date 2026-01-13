#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
uv run python -m src.ft.train_sft_unsloth
