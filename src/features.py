from typing import Any

import numpy as np
import tensorflow as tf


def extract_scalar_features(preprocessed_image: np.ndarray) -> dict[str, float]:
    """Extract simple scalar features from the preprocessed image tensor.

    The tensor is the exact input the model sees (after preprocess), so these
    features can be reproduced for the reference dataset in Chapter 4.2.
    """
    pixels = np.asarray(preprocessed_image)
    return {
        "image_mean": float(np.mean(pixels)),
        "image_std": float(np.std(pixels)),
        "image_min": float(np.min(pixels)),
        "image_max": float(np.max(pixels)),
    }


def build_embedding_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Return a model that outputs the last hidden layer's activations."""
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output,
    )


def extract_embedding(
    embedding_extractor: tf.keras.Model,
    preprocessed_image: np.ndarray,
) -> list[float]:
    """Extract the embedding vector from the model's last hidden layer.

    The embedding captures the representation the classifier learned right before
    the final softmax decision. It is a much stronger drift signal than raw pixel
    statistics.

    The input is the exact 4D (B, H, W, C) tensor returned by preprocess. Errors
    are intentionally not caught here: the caller computes every feature up front
    and guards the whole monitoring block, so a failure skips the whole record
    rather than writing a partial one.
    """
    embedding = embedding_extractor.predict(preprocessed_image, verbose=0).squeeze()
    return embedding.tolist()


def extract_prediction_stats(prediction_result: dict[str, Any]) -> dict[str, Any]:
    """Extract prediction-distribution stats from the postprocess output."""
    probabilities = prediction_result.get("probabilities", {})
    if not probabilities:
        return {
            "predicted_label": prediction_result.get("prediction"),
            "confidence": 0.0,
            "entropy": 0.0,
        }

    probs = np.array(list(probabilities.values()), dtype=float)
    confidence = float(np.max(probs))
    # Shannon entropy (nats); the scale is irrelevant for drift detection.
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))

    return {
        "predicted_label": prediction_result["prediction"],
        "confidence": confidence,
        "entropy": entropy,
    }
