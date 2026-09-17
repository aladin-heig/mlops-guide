import json
import sys
from pathlib import Path
from typing import List

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from utils.seed import set_seed


def get_preview_plot(ds: tf.data.Dataset, labels: List[str]) -> plt.Figure:
    """Plot a preview of the prepared dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds.take(1):
        for ax, image, label_idx in zip(axes.ravel(), images, label_idxs):
            ax.imshow(image.numpy().astype("uint8"), cmap="gray")
            ax.set_title(labels[label_idx.numpy()])
            ax.set_xticks([])
            ax.set_yticks([])

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 prepare.py <raw-dataset-folder> <prepared-dataset-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]

    raw_dataset_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    seed = prepare_params["seed"]
    split = prepare_params["split"]
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    batch_size = prepare_params["batch_size"]

    # Set seed for reproducibility
    set_seed(seed)

    # Read data
    ds_train, ds_val = keras.utils.image_dataset_from_directory(
        raw_dataset_folder,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale" if grayscale else "rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=split,
        subset="both",
    )
    labels = ds_train.class_names

    prepared_dataset_folder.mkdir(parents=True, exist_ok=True)

    # Save the preview plot
    preview_plot = get_preview_plot(ds_train, labels)
    preview_plot.savefig(prepared_dataset_folder / "preview.png")

    # Normalize the data
    normalization_layer = keras.layers.Rescaling(1.0 / 255)
    ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
    ds_val = ds_val.map(lambda x, y: (normalization_layer(x), y))

    # Save the prepared dataset
    with open(prepared_dataset_folder / "labels.json", "w") as f:
        json.dump(labels, f)
    tf.data.Dataset.save(ds_train, str(prepared_dataset_folder / "train"))
    tf.data.Dataset.save(ds_val, str(prepared_dataset_folder / "val"))

    print(f"\nDataset saved at {prepared_dataset_folder.absolute()}")


if __name__ == "__main__":
    main()
