#!/usr/bin/env bash
set -euo pipefail

# Packs the template submission into submission/submission.zip
# Usage: bash examples/template/pack_submission.sh <output_dir>

OUTPUT_DIR="${1:-submission/}"
mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uvx rpzip -r "$(realpath "${OLDPWD}/${OUTPUT_DIR}")/submission.zip" ./*
