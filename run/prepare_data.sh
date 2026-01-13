#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
uv run python -m src.ft.prepare_sft_data
