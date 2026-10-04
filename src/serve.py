from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import bentoml
from bentoml.validators import ContentType
from PIL.Image import Image
from pydantic import Field

from features import (
    build_embedding_extractor,
    extract_embedding,
    extract_prediction_stats,
    extract_scalar_features,
)

# Anchor monitoring output to the project root
LOG_PATH = Path(__file__).resolve().parent.parent / "logs"


@bentoml.service(
    name="celestial_bodies_classifier",
    monitoring={
        "enabled": True,
        "type": "default",
        "options": {"log_path": str(LOG_PATH)},
    },
)
class CelestialBodiesClassifierService:
    bento_model = bentoml.keras.get("celestial_bodies_classifier_model")

    def __init__(self) -> None:
        self.preprocess = self.bento_model.custom_objects["preprocess"]
        self.postprocess = self.bento_model.custom_objects["postprocess"]
        self.model = self.bento_model.load_model()
        self.embedding_extractor = build_embedding_extractor(self.model)

    @bentoml.api()
    def predict(
        self,
        image: Annotated[Image, ContentType("image/jpeg")] = Field(
            description="Planet image to analyze"
        ),
    ) -> Annotated[str, ContentType("application/json")]:
        image = self.preprocess(image)
        predictions = self.model.predict(image)
        result = self.postprocess(predictions)
        self.monitor(image, result)
        return json.dumps(result)

    def monitor(self, preprocessed_image, prediction_result: dict) -> None:
        """Log extracted features for drift monitoring via BentoML's native monitor.

        Every value is computed before the monitoring context is opened, and the
        whole block is guarded: a failure logs to stdout and writes no record, so
        monitoring can never break a prediction nor emit a partial record.
        """
        try:
            scalar_features = extract_scalar_features(preprocessed_image)
            embedding = extract_embedding(self.embedding_extractor, preprocessed_image)
            stats = extract_prediction_stats(prediction_result)

            with bentoml.monitor("celestial_bodies_classifier") as mon:
                for name, value in scalar_features.items():
                    mon.log(value, name=name, role="feature", data_type="numerical")
                mon.log(
                    stats["confidence"],
                    name="confidence",
                    role="feature",
                    data_type="numerical",
                )
                mon.log(
                    stats["entropy"],
                    name="entropy",
                    role="feature",
                    data_type="numerical",
                )
                mon.log(
                    embedding,
                    name="embedding",
                    role="feature",
                    data_type="numerical_sequence",
                )
                mon.log(
                    stats["predicted_label"],
                    name="predicted_label",
                    role="prediction",
                    data_type="categorical",
                )
        except Exception as exc:
            print(f"[monitoring] Failed to log prediction: {exc}")
