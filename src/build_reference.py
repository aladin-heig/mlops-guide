import sys
from pathlib import Path

import bentoml
import pandas as pd
import tensorflow as tf

from features import (
    build_embedding_extractor,
    extract_prediction_stats,
    extract_scalar_features,
)


def main() -> None:
    if len(sys.argv) != 4:
        print("Arguments error. Usage:\n")
        print(
            "\tpython3 build_reference.py <prepared-dataset-folder> <model-folder> <output-parquet>\n"
        )
        exit(1)

    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path(sys.argv[2])
    output_parquet = Path(sys.argv[3])

    # Load training set
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))

    # Import the model to the local BentoML store
    try:
        bentoml.models.import_model(
            f"{model_folder.absolute()}/celestial_bodies_classifier_model.bentomodel"
        )
    except bentoml.exceptions.BentoMLException:
        print("Model already exists in the model store - skipping import.")

    bento_model = bentoml.keras.get("celestial_bodies_classifier_model")
    postprocess = bento_model.custom_objects["postprocess"]
    model = bentoml.keras.load_model("celestial_bodies_classifier_model")

    # Build the embedding extractor without calling the model first.
    feature_extractor = build_embedding_extractor(model)

    records = []
    for images, labels in ds_train:
        # Predict on the whole batch, then process each example individually.
        logits = model.predict(images, verbose=0)
        embeddings = feature_extractor.predict(images, verbose=0)
        for i in range(images.shape[0]):
            image_batch = images[i : i + 1]
            result = postprocess(logits[i : i + 1])
            embedding = embeddings[i]
            prediction_stats = extract_prediction_stats(result)
            record = {
                **extract_scalar_features(image_batch),
                "predicted_label": prediction_stats["predicted_label"],
                "confidence": prediction_stats["confidence"],
                "entropy": prediction_stats["entropy"],
                **{f"emb_{j}": float(embedding[j]) for j in range(embedding.shape[0])},
            }
            records.append(record)

    df = pd.DataFrame(records)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    print(f"\nReference dataset saved at {output_parquet.absolute()}")
    print(f"Rows: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
