import sys
from pathlib import Path
from typing import Tuple

import keras
import numpy as np
import tensorflow as tf
import yaml

from utils.seed import set_seed


def get_model(
    image_shape: Tuple[int, int, int],
    conv_size: int,
    dense_size: int,
    output_classes: int,
) -> keras.Model:
    """Create a simple CNN model"""
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=image_shape),
            keras.layers.Conv2D(conv_size, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((3, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(dense_size, activation="relu"),
            keras.layers.Dense(output_classes),
        ]
    )
    return model


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <prepared-dataset-folder> <model-folder>\n")
        exit(1)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]

    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path(sys.argv[2])

    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    seed = train_params["seed"]
    lr = train_params["lr"]
    epochs = train_params["epochs"]
    conv_size = train_params["conv_size"]
    dense_size = train_params["dense_size"]
    output_classes = train_params["output_classes"]

    # Set seed for reproducibility
    set_seed(seed)

    # Load the prepared datasets and shuffle the training set at each epoch
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_train = ds_train.shuffle(
        buffer_size=ds_train.cardinality(), seed=seed, reshuffle_each_iteration=True
    )
    ds_val = tf.data.Dataset.load(str(prepared_dataset_folder / "val"))

    # Define the model
    model = get_model(image_shape, conv_size, dense_size, output_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()

    # Train the model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_val,
    )

    # Save the model
    model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder.absolute() / "model.keras"
    model.save(model_path)

    # Save the model history
    np.save(model_folder.absolute() / "history.npy", model.history.history)

    print(f"\nModel saved at {model_folder.absolute()}")


if __name__ == "__main__":
    main()
