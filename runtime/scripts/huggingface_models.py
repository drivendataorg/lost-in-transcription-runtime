#!/usr/bin/env python3
"""Manage Hugging Face models listed in runtime/huggingface_models.txt.

Model repo IDs are stable; versions are Hub git commit SHAs. Sync uploads a
model only when it is not yet in Azure (stores the SHA in
``{model_id}/.hf_revision``). Existing Azure models are never overwritten so
running pods keep a stable view; if the Hub revision drifts, sync logs a
warning for a competition administrator to refresh manually when safe.

Usage:
  huggingface_models.py download [--output DIR] [MODEL_ID ...]
  huggingface_models.py verify
  huggingface_models.py sync ACCOUNT CONTAINER [--output DIR]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

MANIFEST = Path("runtime/huggingface_models.txt")
REVISION_BLOB = ".hf_revision"
DEFAULT_OUTPUT = Path(".huggingface_models")
UPLOAD_MAX_CONCURRENCY = 8
# Soft size check in verify only; sync still uploads larger models.
SIZE_WARN_BYTES = 100 * 1024**3


def read_models(manifest: Path = MANIFEST) -> list[str]:
    """Return model IDs from the manifest. Lines starting with # are comments."""
    models: list[str] = []
    for raw_line in manifest.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        models.append(line)
    return models


def hub_revision(model_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def download_model(output: Path, model_id: str, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    destination = output / model_id
    label = f"{model_id}" + (f" @ {revision}" if revision else "")
    print(f"Downloading {label} to {destination}", flush=True)
    snapshot_download(repo_id=model_id, revision=revision, local_dir=destination)
    # local_dir keeps a small hub cache; don't upload it to Azure.
    shutil.rmtree(destination / ".cache", ignore_errors=True)
    return destination


def cleanup_model(output: Path, model_id: str) -> None:
    model_dir = output / model_id
    shutil.rmtree(model_dir, ignore_errors=True)
    # Remove empty parents left under output (e.g. org/ for org/model).
    parent = model_dir.parent
    while parent != output and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def blob_service(account: str):
    """Blob client using DefaultAzureCredential (works with azure/login + az CLI)."""
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )


def azure_revision(account: str, container: str, model_id: str) -> str | None:
    """Return the Hub SHA stored in Azure, or None if missing."""
    from azure.core.exceptions import ResourceNotFoundError

    blob = blob_service(account).get_blob_client(
        container=container,
        blob=f"{model_id}/{REVISION_BLOB}",
    )
    try:
        return blob.download_blob().readall().decode("utf-8").strip() or None
    except ResourceNotFoundError:
        return None


def azure_upload_model(local_dir: Path, account: str, container: str, model_id: str) -> None:
    """Upload a local model directory to Azure (never overwrites existing blobs)."""
    container_client = blob_service(account).get_container_client(container)
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    print(f"Uploading {len(files)} files for {model_id} to Azure...", flush=True)
    for path in files:
        blob_name = f"{model_id}/{path.relative_to(local_dir).as_posix()}"
        with path.open("rb") as handle:
            container_client.upload_blob(
                name=blob_name,
                data=handle,
                overwrite=False,
                max_concurrency=UPLOAD_MAX_CONCURRENCY,
            )


def append_step_summary(line: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def warn(message: str) -> None:
    """Print a warning; use a GitHub Actions annotation when running in CI."""
    print(f"WARNING: {message}", flush=True)
    # Visible in the Actions UI without failing the job.
    print(f"::warning::{message}", flush=True)
    append_step_summary(f"- warning: {message}")


def cmd_download(output: Path, model_ids: list[str] | None) -> None:
    """Download models listed in the manifest or given by model_ids.

    Only downloads models locally; does not upload them to Azure.
    """
    models = model_ids or read_models()
    if not models:
        print(f"No Hugging Face models listed in {MANIFEST}.")
        return
    output.mkdir(parents=True, exist_ok=True)
    for model_id in models:
        download_model(output, model_id)


def cmd_verify() -> None:
    """Confirm each manifest model exists on the Hub and is downloadable (no weight fetch).

    Warns when a model's total Hub file size exceeds 100 GB, but does not fail.
    """
    models = read_models()
    if not models:
        print(f"No Hugging Face models listed in {MANIFEST}.")
        return

    from huggingface_hub import snapshot_download

    for model_id in models:
        print(f"=== Verifying {model_id} is downloadable on the Hub ===", flush=True)
        revision = hub_revision(model_id)
        # dry_run resolves the repo/file list without fetching weights.
        files = snapshot_download(repo_id=model_id, revision=revision, dry_run=True)
        total_bytes = sum(getattr(f, "file_size", 0) or 0 for f in files)
        size_gb = total_bytes / (1024**3)
        print(f"OK {model_id} @ {revision} ({size_gb:.1f} GB across {len(files)} files)", flush=True)
        append_step_summary(
            f"- `{model_id}` @ `{revision}`: downloadable ({size_gb:.1f} GB, {len(files)} files)"
        )
        if total_bytes > SIZE_WARN_BYTES:
            warn(
                f"{model_id} is {size_gb:.1f} GB (> {SIZE_WARN_BYTES // (1024**3)} GB). "
                f"There is no hard cap, but confirm this size is intentional before syncing to Azure."
            )


def cmd_sync(account: str, container: str, output: Path) -> None:
    """Upload models listed in the manifest to Azure when missing; never overwrite."""
    models = read_models()
    if not models:
        print(f"No Hugging Face models listed in {MANIFEST}.")
        return

    output.mkdir(parents=True, exist_ok=True)
    for model_id in models:
        print(f"=== Syncing {model_id} ===", flush=True)
        desired_sha = hub_revision(model_id)
        print(f"Hub revision: {desired_sha}", flush=True)

        remote_sha = azure_revision(account, container, model_id)
        print(f"Azure revision: {remote_sha or '(none)'}", flush=True)
        if remote_sha is not None:
            if remote_sha == desired_sha:
                print(
                    f"Skipping {model_id}; Azure already has this Hub revision.",
                    flush=True,
                )
            else:
                warn(
                    f"{model_id} is already in Azure at revision {remote_sha}, but the "
                    f"Hub now has {desired_sha}. Sync will not overwrite Azure blobs "
                    f"while pods may be reading them. Contact a competition "
                    f"administrator to refresh the model manually when no pods are running."
                )
            continue

        destination = download_model(output, model_id, revision=desired_sha)
        (destination / REVISION_BLOB).write_text(desired_sha + "\n", encoding="utf-8")
        azure_upload_model(destination, account, container, model_id)
        cleanup_model(output, model_id)
        print(f"Uploaded {model_id} @ {desired_sha}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download model snapshots locally.")
    download.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT}).",
    )
    download.add_argument(
        "models",
        nargs="*",
        help="Optional model IDs (default: all in the manifest).",
    )

    sub.add_parser("verify", help="Check manifest models exist and are downloadable.")

    sync = sub.add_parser(
        "sync",
        help="Upload new models to Azure; never overwrite existing ones.",
    )
    sync.add_argument("account", help="Azure storage account name.")
    sync.add_argument("container", help="Azure blob container name.")
    sync.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Scratch directory for downloads (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args(argv)

    if args.command == "download":
        cmd_download(args.output, args.models or None)
    elif args.command == "verify":
        cmd_verify()
    elif args.command == "sync":
        cmd_sync(args.account, args.container, args.output)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
