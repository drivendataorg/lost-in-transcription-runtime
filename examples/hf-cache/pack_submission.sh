#!/usr/bin/env bash
set -euo pipefail

# Packs the hf-cache example submission into submission/submission.zip.
# The zip holds no model files. The example reads the model from the shared
# Hugging Face cache that the pod mounts at /code_execution/huggingface_models.
# Usage: bash examples/hf-cache/pack_submission.sh <output_dir>

OUTPUT_DIR="${1:-submission/}"
mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uvx rpzip -r "$(realpath "${OLDPWD}/${OUTPUT_DIR}")/submission.zip" ./*
