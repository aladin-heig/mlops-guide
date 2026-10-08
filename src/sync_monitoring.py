import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.cloud import storage
from evidently.ui.workspace import Workspace

from monitor import build_dashboard, generate_report, get_or_create_project

BUCKET_NAME = os.environ.get("GCP_BUCKET_NAME")
LOG_PREFIX = "logs/bentoml"
PROJECT_NAME = os.environ.get("EVIDENTLY_PROJECT_NAME", "celestial-bodies-classifier")
WORKSPACE_PREFIX = os.environ.get("EVIDENTLY_WORKSPACE_PREFIX", "evidently-workspace")
LOG_CUTOFF_HOURS = int(os.environ.get("LOG_CUTOFF_HOURS", "24"))
# Keep LOG_CUTOFF_HOURS in sync with the monitoring workflow schedule. If the
# workflow runs once a day, a 24-hour lookback covers one run's worth of logs.


def download_latest_logs(bucket_name: str, prefix: str, dest: Path) -> None:
    """Download log objects from the last N hours into a directory of log files."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOG_CUTOFF_HOURS)
    blobs = [
        blob for blob in bucket.list_blobs(prefix=prefix) if blob.updated >= cutoff
    ]

    if not blobs:
        print(
            f"No log objects found under gs://{bucket_name}/{prefix} "
            f"in the last {LOG_CUTOFF_HOURS} hours"
        )
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    for i, blob in enumerate(sorted(blobs, key=lambda b: b.updated)):
        out_path = dest / f"data.{i + 1}.log"
        blob.download_to_filename(out_path)


def main() -> None:
    if not BUCKET_NAME:
        print("GCP_BUCKET_NAME environment variable is required")
        sys.exit(1)

    reference_path = Path("data/reference_features.parquet")
    if not reference_path.exists():
        print(f"Reference dataset not found at {reference_path}. Run dvc pull first.")
        sys.exit(1)

    output_dir = Path("monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "logs" / "celestial_bodies_classifier" / "data"
        download_latest_logs(BUCKET_NAME, LOG_PREFIX, log_dir)
        snapshot = generate_report(reference_path, log_dir, output_dir)

    workspace = Workspace.create(f"gs://{BUCKET_NAME}/{WORKSPACE_PREFIX}")
    project = get_or_create_project(workspace, PROJECT_NAME)
    workspace.add_run(project.id, snapshot, include_data=False)
    build_dashboard(project, snapshot)
    print(f"Snapshot added to project {project.name} (ID: {project.id})")


if __name__ == "__main__":
    main()
