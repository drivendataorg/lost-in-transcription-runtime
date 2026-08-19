#!/usr/bin/env bash

set -euxo pipefail

log() {
    set +x
    local level="$1"; shift
    printf '%s | %-4s | %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S.%3N')" \
        "$level" \
        "$*"
    set -x
}

main () {
    if [ -n "${STARTUP_SLEEP_SECONDS:-}" ]; then
        log INFO "Sleeping ${STARTUP_SLEEP_SECONDS} seconds before running submission..."
        sleep "${STARTUP_SLEEP_SECONDS}"
    fi

    expected_filename=main.py

    cd /code_execution

    if [ "${LOST_IN_TRANSCRIPTION_IS_SMOKE:-0}" -eq 1 ]; then
        log INFO "This is a smoke test run."
    fi

    # Check that expected entrypoint script exists
    submission_files=$(zip -sf ./submission/submission.zip)
    if ! grep -F -q -- "$expected_filename" <<<"$submission_files"; then
        log ERROR "Submission zip archive must include $expected_filename";
        return 1;
    fi

    log INFO "Unpacking submission into src/..."
    unzip ./submission/submission.zip -d ./src

    log INFO "Showing current working directory contents:"
    ls -alh

    log INFO "Showing src/ directory contents:"
    find src/

    log INFO "Running submission..."

    uv run src/main.py

    # Locate the expected submission file; copy to submission/ if needed.
    if [ -f "./submission/submission.csv" ]; then
        log INFO "Found submission.csv in submission/."
    elif [ -f "./src/submission.csv" ]; then
        log INFO "Copying submission.csv from src/ to submission/."
        cp "./src/submission.csv" "./submission/submission.csv"
    else
        log ERROR "Script did not produce a submission.csv file."
        return 1
    fi
}

main |& tee "/code_execution/submission/log.txt"
exit_code=${PIPESTATUS[0]}

cp /code_execution/submission/log.txt /tmp/log

log INFO "Submission run completed with exit code: $exit_code"

exit $exit_code
