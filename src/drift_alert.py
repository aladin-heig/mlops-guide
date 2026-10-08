import json
import os
import subprocess
from pathlib import Path

REPORT_PATH = Path(os.environ.get("REPORT_PATH", "monitoring/report.json"))
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "")
ISSUE_LABEL = "drift-alert"


def extract_alerts(report_data: dict) -> list[str]:
    """Return a list of human-readable alert lines from an Evidently report."""
    alerts: list[str] = []

    for metric in report_data.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")
        config = metric.get("config", {})

        if "ValueDrift" in name:
            score = value if isinstance(value, (int, float)) else None
            threshold = config.get("threshold")
            if score is not None and threshold is not None and score > threshold:
                column = config.get("column", "unknown")
                alerts.append(f"{column}: {score:.4f} > {threshold:.4f}")

        elif "EmbeddingsDrift" in name:
            score = value if isinstance(value, (int, float)) else None
            threshold = config.get("drift_method", {}).get("threshold")
            if score is not None and threshold is not None and score > threshold:
                embeddings_name = config.get("embeddings_name", "embedding")
                alerts.append(f"{embeddings_name}: {score:.4f} > {threshold:.4f}")

        elif "DriftedColumnsCount" in name:
            share = value.get("share") if isinstance(value, dict) else None
            threshold = config.get("drift_share")
            if share is not None and threshold is not None and share > threshold:
                alerts.append(f"drifted columns share: {share:.4f} > {threshold:.4f}")

    return alerts


def open_drift_alert_issue_exists() -> bool:
    """Return True if an open drift-alert issue already exists."""
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--label",
            ISSUE_LABEL,
            "--state",
            "open",
            "--json",
            "number",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(json.loads(result.stdout))


def create_issue(body: str) -> None:
    """Open a GitHub issue with the given Markdown body."""
    subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--title",
            "Drift detected",
            "--body",
            body,
            "--label",
            ISSUE_LABEL,
        ],
        check=True,
    )


def build_issue_body(alerts: list[str]) -> str:
    """Build the Markdown body for a drift-alert GitHub issue."""
    lines = [
        "## Drift alert",
        "",
        "The monitoring workflow detected drift in production predictions.",
        "",
    ]
    lines.extend(f"- {alert}" for alert in alerts)

    if DASHBOARD_URL:
        lines.extend(["", f"[View dashboard]({DASHBOARD_URL})"])

    lines.extend(
        [
            "",
            "### Next steps",
            "",
            "- Review this alert to decide whether to roll back, label new "
            "data, or dismiss and keep monitoring.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    report_data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    alerts = extract_alerts(report_data)

    if not alerts:
        print("No drift alert")
        return

    if open_drift_alert_issue_exists():
        print("An open drift-alert issue already exists; skipping")
        return

    create_issue(build_issue_body(alerts))
    print("Drift alert issue created")


if __name__ == "__main__":
    main()
