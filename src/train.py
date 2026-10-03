import json
import sys
from pathlib import Path
from typing import Tuple

import bentoml
import keras
import numpy as np
import tensorflow as tf
import yaml
from PIL.Image import Image

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

    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

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

    def preprocess(x: Image):
        # convert PIL image to tensor
        x = x.convert("L" if grayscale else "RGB")
        x = x.resize(image_size)
        x = np.array(x, dtype=np.float32) / 255.0
        # add channel dimension for grayscale
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)
        # add batch dimension
        x = np.expand_dims(x, axis=0)
        return x

    def postprocess(x: Image):
        return {
            "prediction": labels[tf.argmax(x, axis=-1).numpy()[0]],
            "probabilities": {
                labels[i]: prob
                for i, prob in enumerate(tf.nn.softmax(x).numpy()[0].tolist())
            },
        }

    # Save the model using BentoML to its model store
    bentoml.keras.save_model(
        "celestial_bodies_classifier_model",
        model,
        include_optimizer=True,
        custom_objects={
            "preprocess": preprocess,
            "postprocess": postprocess,
        },
    )

    # Export the model from the model store to the local model folder
    bentoml.models.export_model(
        "celestial_bodies_classifier_model:latest",
        f"{model_folder.absolute()}/celestial_bodies_classifier_model.bentomodel",
    )

    # Save the model history
    np.save(model_folder.absolute() / "history.npy", model.history.history)

    print(f"\nModel saved at {model_folder.absolute()}")


if __name__ == "__main__":
    main()
