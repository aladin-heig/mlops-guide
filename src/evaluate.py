import json
import sys
from pathlib import Path
from typing import List

import bentoml
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def get_training_plot(model_history: dict) -> plt.Figure:
    """Plot the training and validation loss"""
    epochs = range(1, len(model_history["loss"]) + 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, model_history["loss"], label="Training loss")
    plt.plot(epochs, model_history["val_loss"], label="Validation loss")
    plt.xticks(epochs)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    return fig


def get_pred_preview_plot(
    model: keras.Model, ds_val: tf.data.Dataset, labels: List[str]
) -> plt.Figure:
    """Plot a preview of the predictions"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds_val.take(1):
        pred_idxs = np.argmax(model.predict(images, verbose=0), axis=1)
        for ax, image, true_idx, pred_idx in zip(
            axes.ravel(), images, label_idxs, pred_idxs
        ):
            true_label = labels[true_idx.numpy()]
            pred_label = labels[pred_idx]
            ax.imshow(image.numpy().squeeze(), cmap="gray")
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.set_xticks([])
            ax.set_yticks([])

            border_color = "lime" if true_idx.numpy() == pred_idx else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)

    return fig


def get_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> plt.Figure:
    """Plot the confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        normalize="true",
        cmap="Blues",
        values_format=".2f",
        ax=ax,
        colorbar=True,
    )

    for value, text in zip(display.confusion_matrix.ravel(), display.text_.ravel()):
        text.set_fontsize(7)
        if np.isclose(value, 0.0):
            text.set_color("lightgray")

    ax.set_xticklabels(labels, rotation=90)
    ax.set_title("Validation confusion matrix")

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 evaluate.py <model-folder> <prepared-dataset-folder>\n")
        exit(1)

    model_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    evaluation_folder = Path("evaluation")
    plots_folder = Path("plots")

    # Create folders
    (evaluation_folder / plots_folder).mkdir(parents=True, exist_ok=True)

    # Load files
    ds_val = tf.data.Dataset.load(str(prepared_dataset_folder / "val"))
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Import the model to the model store from a local model folder
    try:
        bentoml.models.import_model(
            f"{model_folder.absolute()}/celestial_bodies_classifier_model.bentomodel"
        )
    except bentoml.exceptions.BentoMLException:
        print("Model already exists in the model store - skipping import.")

    # Load model
    model = bentoml.keras.load_model("celestial_bodies_classifier_model")
    model_history = np.load(
        model_folder.absolute() / "history.npy", allow_pickle=True
    ).item()

    # Log metrics
    val_loss, val_acc = model.evaluate(ds_val)
    preds = model.predict(ds_val)
    y_true = tf.concat([y for _, y in ds_val], axis=0).numpy()
    y_pred = np.argmax(preds, axis=1)

    metrics = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    print(f"Validation loss: {metrics['val_loss']:.2f}")
    print(f"Validation accuracy: {metrics['val_acc'] * 100:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1 score:  {metrics['f1_score']:.2f}")

    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # Save training history plot
    fig = get_training_plot(model_history)
    fig.savefig(evaluation_folder / plots_folder / "training_history.png")

    # Save predictions preview plot
    fig = get_pred_preview_plot(model, ds_val, labels)
    fig.savefig(evaluation_folder / plots_folder / "pred_preview.png")

    # Save confusion matrix plot
    fig = get_confusion_matrix_plot(y_true, y_pred, labels)
    fig.savefig(evaluation_folder / plots_folder / "confusion_matrix.png")

    print(
        f"\nEvaluation metrics and plot files saved at {evaluation_folder.absolute()}"
    )


if __name__ == "__main__":
    main()
