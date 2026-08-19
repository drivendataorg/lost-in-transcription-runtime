set dotenv-load

block_internet := "true"

IMAGE_NAME := "lost-in-transcription-competition"
LOCAL_RUNTIME_IMAGE_REF := f"local.invalid/{{IMAGE_NAME}}:dev"
LOCAL_TEST_IMAGE_REF := f"local.invalid/{{IMAGE_NAME}}:test"
RUNTIME_IMAGE_REF := f"lostintranscriptionprodacr.azurecr.io/{{IMAGE_NAME}}:latest"

MOUNT_DATA := f"--mount type=bind,source={{justfile_directory()}}/data/data,target=/code_execution/data,readonly"
MOUNT_SUBMISSION := f"--mount type=bind,source={{justfile_directory()}}/submission/,target=/code_execution/submission"
NETWORK_ARGS := if block_internet == "true" { "--network none" } else { "" }
GPU_ARGS := `if nvidia-smi > /dev/null 2>&1; then echo "--gpus all"; else echo ""; fi`

# Print this help documentation
help:
    @echo "Lost In Transcription Runtime Justfile"
    @echo ""
    @echo "---"
    @just --list

# Print active variables for debugging
debug:
    @echo "block_internet={{block_internet}}"
    @echo "RUNTIME_IMAGE_REF={{RUNTIME_IMAGE_REF}}"
    @echo "LOCAL_RUNTIME_IMAGE_REF={{LOCAL_RUNTIME_IMAGE_REF}}"
    @echo "LOCAL_TEST_IMAGE_REF={{LOCAL_TEST_IMAGE_REF}}"
    @echo "MOUNT_DATA={{MOUNT_DATA}}"
    @echo "MOUNT_SUBMISSION={{MOUNT_SUBMISSION}}"
    @echo "NETWORK_ARGS={{NETWORK_ARGS}}"

## Local testing

# Pull official image
[group('* test submission locally')]
pull:
    docker pull "{{RUNTIME_IMAGE_REF}}"

[confirm("Are you sure you want to overwrite the existing submission/submission.zip file? (y/n)")]
_confirm_submission_overwrite:
    @echo "Overwriting existing submission/submission.zip file."
    rm -f "{{justfile_directory()}}/submission/submission.zip"

# Prompt to overwrite only if submission/submission.zip already exists.
_maybe_confirm_overwrite:
    @if [ -f "{{justfile_directory()}}/submission/submission.zip" ]; then \
        just _confirm_submission_overwrite; \
    fi

# Pack submission_src/ into submission/submission.zip
[group('* test submission locally')]
pack-submission:
    #!/usr/bin/env bash
    if [ -z "$(ls -A submission_src)" ]; then
        echo "ERROR: submission_src/ directory is empty. Cannot create submission.zip with no files.";
        exit 1
    fi
    just _maybe_confirm_overwrite
    cd submission_src && uvx rpzip -r "{{justfile_directory()}}/submission/submission.zip" ./*

# Check contents of current submission.zip file
[group('* test submission locally')]
check-submission:
    @if unzip -Z1 submission/submission.zip | grep -Fxq main.py; then \
        echo "VALIDATION PASSED: Submission ZIP archive contains main.py."; \
    else \
        echo "ERROR: Submission ZIP archive must include main.py in the root directory."; \
        exit 1; \
    fi
    unzip -l submission/submission.zip

# Run submission with official image
[group('* test submission locally')]
run:
    docker run \
        --rm \
        {{MOUNT_DATA}} \
        {{MOUNT_SUBMISSION}} \
        {{NETWORK_ARGS}} \
        "{{RUNTIME_IMAGE_REF}}"

# Run official image with interactive shell
[group('* test submission locally')]
interact:
    docker run -it \
        --rm \
        {{MOUNT_DATA}} \
        {{MOUNT_SUBMISSION}} \
        {{NETWORK_ARGS}} \
        "{{RUNTIME_IMAGE_REF}}" \
        bash

## Dev

# Lock dependencies
[group('development')]
[working-directory('runtime')]
lock *ARGS:
    uv lock {{ARGS}}

# Check whether lockfile is up to date
[group('development')]
[working-directory('runtime')]
lock-check:
    uv lock --check

# Build local dev runtime image
[group('development')]
build *ARGS:
    docker build runtime/ --platform=linux/amd64 --target runtime --tag "{{LOCAL_RUNTIME_IMAGE_REF}}" {{ARGS}}

# Run submission with local dev runtime image
[group('development')]
dev-run:
    docker run \
        --rm \
        {{MOUNT_DATA}} \
        {{MOUNT_SUBMISSION}} \
        {{NETWORK_ARGS}} \
        "{{LOCAL_RUNTIME_IMAGE_REF}}"

# Run local dev runtime image with interactive shell
[group('development')]
dev-interact:
    docker run -it \
        --rm \
        {{MOUNT_DATA}} \
        {{MOUNT_SUBMISSION}} \
        {{NETWORK_ARGS}} \
        "{{LOCAL_RUNTIME_IMAGE_REF}}" \
        bash

## Test image

# Build local test runtime image
[group('development')]
test-build:
    docker build runtime/ --platform=linux/amd64 --target test-runtime --tag "{{LOCAL_TEST_IMAGE_REF}}"

# Run tests with local test runtime image
[group('development')]
test-run:
    docker run \
        --rm \
        {{GPU_ARGS}} \
        {{NETWORK_ARGS}} \
        "{{LOCAL_TEST_IMAGE_REF}}"

# Run local test runtime image with interactive shell
[group('development')]
test-interact:
    docker run -it \
        --rm \
        {{GPU_ARGS}} \
        {{NETWORK_ARGS}} \
        "{{LOCAL_TEST_IMAGE_REF}}" \
        bash

## Examples

# Pack example submission into a submission.zip file
[group('* test submission locally')]
pack-example EXAMPLE_NAME: _maybe_confirm_overwrite
    bash examples/{{EXAMPLE_NAME}}/pack_submission.sh submission/

## Scoring

# Score a submission CSV against a ground truth CSV using MDC-normalized WER
[group('* test submission locally')]
score ACTUAL PREDICTED=(justfile_directory() / "submission" / "submission.csv"):
    uv run "{{justfile_directory()}}/score.py" "{{ACTUAL}}" --predicted-path "{{PREDICTED}}"
