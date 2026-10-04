import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import EmbeddingsDrift
from evidently.metrics.embeddings import ModelDriftMethod
from evidently.presets import DataDriftPreset
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import counter_panel, line_plot_panel, text_panel
from evidently.ui.workspace import Workspace

# Resolve everything against the project root (the directory above src/) so the
# script works no matter which directory it is launched from. serve.py anchors
# the monitoring log_path to this same location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATH = PROJECT_ROOT / "data/reference_features.parquet"
# BentoML native monitoring writes rotating JSONL files to
# <log_path>/<monitor_name>/data/*.log; log_path is set in serve.py.
MONITOR_NAME = "celestial_bodies_classifier"
LOG_DIR = PROJECT_ROOT / "logs" / MONITOR_NAME / "data"
WORKSPACE_PATH = PROJECT_ROOT / "monitoring/workspace"
REPORT_PATH = PROJECT_ROOT / "monitoring"
PROJECT_NAME = "celestial-bodies-classifier"

# Features monitored for drift.
# image_min is excluded: black background makes it zero-variance (no drift signal)
SCALAR_COLUMNS = [
    "image_mean",
    "image_std",
    "image_max",
    "confidence",
    "entropy",
]
CATEGORICAL_COLUMNS = ["predicted_label"]
EMBEDDING_NAME = "image_embedding"
DRIFT_COLUMNS = SCALAR_COLUMNS + CATEGORICAL_COLUMNS

# Drift detection thresholds and methods.
DRIFT_SHARE_THRESHOLD = 0.5
NUM_DRIFT_METHOD = "wasserstein"
NUM_DRIFT_THRESHOLD = 0.3
CAT_DRIFT_METHOD = "jensenshannon"
CAT_DRIFT_THRESHOLD = 0.1
EMBEDDING_DRIFT_THRESHOLD = 0.7


def expand_embedding_column(df: pd.DataFrame) -> pd.DataFrame:
    """Unpack a list-valued embedding column into one column per dimension.

    Evidently's embedding drift metric expects embeddings as separate columns.
    """
    if "embedding" not in df.columns:
        return df

    df = df.dropna(subset=["embedding"]).copy()
    if df.empty:
        return df.drop(columns=["embedding"])

    embedding_dim = len(df["embedding"].iloc[0])
    emb_columns = [f"emb_{i}" for i in range(embedding_dim)]
    emb_df = pd.DataFrame(df["embedding"].tolist(), columns=emb_columns, index=df.index)
    return pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)


def get_or_create_project(workspace: Workspace, name: str):
    """Return an existing project by name or create a new one."""
    for project in workspace.search_project(name):
        if project.name == name:
            return project
    return workspace.create_project(
        name=name,
        description="Drift monitoring for the celestial bodies classifier",
    )


def drop_constant_embedding_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    emb_columns: list[str],
) -> list[str]:
    """Drop embedding dimensions that are constant in both reference and current.

    ReLU activations often produce exact zeros for some neurons, which gives
    those embedding columns zero variance. Evidently's internal correlation
    calculations divide by the standard deviation and emit noisy
    ``RuntimeWarning`` messages for constant columns. Since a constant column
    carries no information, dropping it is safe and does not change the drift
    signal.
    """
    constant = [
        col
        for col in emb_columns
        if col in reference_df.columns
        and col in current_df.columns
        and reference_df[col].std() == 0
        and current_df[col].std() == 0
    ]

    if constant:
        print(
            f"[info] Dropping {len(constant)} embedding dimension(s) that are "
            "constant zero in both reference and current data: "
            f"{', '.join(constant[:5])}{'...' if len(constant) > 5 else ''}."
        )
        reference_df.drop(columns=constant, inplace=True)
        current_df.drop(columns=constant, inplace=True)

    return [c for c in emb_columns if c not in constant]


def generate_report(
    reference_path: Path,
    log_dir: Path,
    output_dir: Path,
):
    """Build an Evidently drift report from a reference dataset and log files."""
    reference_df = pd.read_parquet(reference_path)
    # Drop any columns that are not part of the drift feature set.
    reference_df = reference_df.drop(columns=["true_label"], errors="ignore")

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    current_df = pd.concat(
        [pd.read_json(f, lines=True) for f in log_files],
        ignore_index=True,
    )
    if current_df.empty:
        raise ValueError(
            f"Log files in {log_dir} contain no rows. "
            "Run the service and make some predictions first."
        )

    current_df = current_df.drop(
        columns=["timestamp", "request_id", "date", "trace_id"],
        errors="ignore",
    )
    current_df = expand_embedding_column(current_df)

    if current_df.empty:
        raise ValueError(
            "Current prediction data is empty after dropping missing embeddings."
        )

    emb_columns = [c for c in reference_df.columns if c.startswith("emb_")]
    emb_columns = drop_constant_embedding_columns(reference_df, current_df, emb_columns)

    data_definition = DataDefinition(
        numerical_columns=SCALAR_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        embeddings={EMBEDDING_NAME: emb_columns},
    )

    reference_data = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_data = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report(
        [
            DataDriftPreset(
                columns=DRIFT_COLUMNS,
                drift_share=DRIFT_SHARE_THRESHOLD,
                num_method=NUM_DRIFT_METHOD,
                num_threshold=NUM_DRIFT_THRESHOLD,
                cat_method=CAT_DRIFT_METHOD,
                cat_threshold=CAT_DRIFT_THRESHOLD,
            ),
            EmbeddingsDrift(
                embeddings_name=EMBEDDING_NAME,
                drift_method=ModelDriftMethod(threshold=EMBEDDING_DRIFT_THRESHOLD),
            ),
        ]
    )
    snapshot = report.run(current_data=current_data, reference_data=reference_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_dir / "report.html"))
    snapshot.save_json(str(output_dir / "report.json"))

    return snapshot


def drift_summary(snapshot) -> str:
    """Return a short human-readable drift summary from the snapshot."""
    for metric in snapshot.dict().get("metrics", []):
        if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            count = value.get("count", 0)
            share = value.get("share", 0)
            return (
                f"Dataset drift detected for {share:.1%} of columns "
                f"({int(count)} out of {len(DRIFT_COLUMNS)})."
            )
    return "No drift summary available."


def build_dashboard(project, snapshot) -> None:
    """Populate the Evidently project dashboard with monitoring panels."""
    project.dashboard.clear_dashboard()
    tab = "Drift"

    project.dashboard.add_panel(
        text_panel(
            title="Celestial Bodies Classifier",
            description="Drift monitoring dashboard",
        ),
        tab=tab,
    )
    project.dashboard.add_panel(
        text_panel(
            title="Data Drift Summary",
            description=drift_summary(snapshot),
        ),
        tab=tab,
    )
    project.dashboard.add_panel(
        counter_panel(
            title="Drifted Columns",
            size="half",
            values=[
                PanelMetric(
                    legend="drifted",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "count"},
                ),
            ],
            aggregation="last",
        ),
        tab=tab,
    )
    project.dashboard.add_panel(
        counter_panel(
            title="Share of Drifted Columns",
            size="half",
            values=[
                PanelMetric(
                    legend="share",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                ),
            ],
            aggregation="last",
        ),
        tab=tab,
    )
    project.dashboard.add_panel(
        line_plot_panel(
            title="Feature Drift (p-value)",
            values=[
                PanelMetric(
                    legend=col,
                    metric="ValueDrift",
                    metric_labels={"column": col},
                )
                for col in DRIFT_COLUMNS
            ],
        ),
        tab=tab,
    )
    project.dashboard.add_panel(
        counter_panel(
            title="Image Embedding Drift",
            size="half",
            values=[
                PanelMetric(
                    legend="drift score",
                    metric="EmbeddingsDrift",
                    metric_labels={"embeddings_name": EMBEDDING_NAME},
                ),
            ],
            aggregation="last",
        ),
        tab=tab,
    )
    project.dashboard.add_panel(
        line_plot_panel(
            title="Image Embedding Drift Over Time",
            size="half",
            values=[
                PanelMetric(
                    legend="drift score",
                    metric="EmbeddingsDrift",
                    metric_labels={"embeddings_name": EMBEDDING_NAME},
                ),
            ],
        ),
        tab=tab,
    )


def main() -> None:
    if not REFERENCE_PATH.exists():
        print(
            f"Reference dataset not found at {REFERENCE_PATH}. "
            "Run build_reference.py first."
        )
        sys.exit(1)

    if not LOG_DIR.exists():
        print(
            f"Prediction log directory not found at {LOG_DIR}. "
            "Run the service and make some predictions first."
        )
        sys.exit(1)

    if not list(LOG_DIR.glob("*.log")):
        print(
            f"Prediction log directory {LOG_DIR} exists but contains no *.log files. "
            "Run the service and make some predictions first."
        )
        sys.exit(1)

    snapshot = generate_report(REFERENCE_PATH, LOG_DIR, REPORT_PATH)

    WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    workspace = Workspace.create(str(WORKSPACE_PATH))
    project = get_or_create_project(workspace, PROJECT_NAME)
    workspace.add_run(project.id, snapshot, include_data=False)
    build_dashboard(project, snapshot)

    print(f"\nSnapshot added to workspace: {WORKSPACE_PATH.absolute()}")
    print(f"Project: {project.name} (ID: {project.id})")

    # Print a concise drift summary from the JSON output
    report_data = snapshot.dict()
    for metric in report_data.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        if "Drift" in metric_name:
            value = metric.get("value", {})
            print(f"{metric_name}: value={value}")


if __name__ == "__main__":
    main()
