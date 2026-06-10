"""
Prediction utilities for the Egyptian artifact recognition API.

The active model is loaded once at import time. Preprocessing is selected from
the loaded model's input shape and whether it already contains preprocessing
layers, so both model.h5 and model_v2_backup.h5 can be inspected safely.
"""

import io
import json
import logging
import os
import shutil
import tempfile
from typing import Any

import h5py
import numpy as np
from PIL import Image
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.h5")
_MODEL_PATH = os.path.join(_MODEL_DIR, _MODEL_FILENAME)
_CLASS_NAMES_PATH = os.path.join(_MODEL_DIR, "class_names.json")


def _patch_legacy_h5_config(value):
    if isinstance(value, dict):
        class_name = value.get("class_name")
        config = value.get("config")

        if class_name == "DTypePolicy" and isinstance(config, dict):
            return config.get("name", "float32")

        if class_name == "InputLayer" and isinstance(config, dict):
            config.pop("optional", None)

        if class_name == "BatchNormalization" and isinstance(config, dict):
            config.pop("renorm", None)
            config.pop("renorm_clipping", None)
            config.pop("renorm_momentum", None)

        if isinstance(config, dict):
            config.pop("quantization_config", None)

            dtype_config = config.get("dtype")
            if isinstance(dtype_config, dict) and dtype_config.get("class_name") == "DTypePolicy":
                dtype_policy_config = dtype_config.get("config") or {}
                config["dtype"] = dtype_policy_config.get("name", "float32")

        return {key: _patch_legacy_h5_config(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_patch_legacy_h5_config(item) for item in value]

    return value


def _load_model_with_compatibility(model_path: str):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as error:
        logger.warning("Standard model load failed, retrying with H5 config compatibility patch: %s", error)

        patched_path = os.path.join(tempfile.gettempdir(), f"patched_{os.path.basename(model_path)}")
        shutil.copyfile(model_path, patched_path)

        with h5py.File(patched_path, "r+") as h5_file:
            raw_config = h5_file.attrs.get("model_config")
            if raw_config is None:
                raise

            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")

            model_config = json.loads(raw_config)
            model_config = _patch_legacy_h5_config(model_config)
            h5_file.attrs.modify("model_config", json.dumps(model_config).encode("utf-8"))

        return tf.keras.models.load_model(patched_path, compile=False)


_model = _load_model_with_compatibility(_MODEL_PATH)

with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    _class_names: list[str] = json.load(f)


def _shape_to_list(shape: Any) -> list[Any]:
    if hasattr(shape, "as_list"):
        return shape.as_list()
    if isinstance(shape, (list, tuple)):
        return list(shape)
    return [shape]


def _model_input_shape() -> list[Any]:
    shape = _model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return _shape_to_list(shape)


def _model_output_shape() -> list[Any]:
    shape = _model.output_shape
    if isinstance(shape, list):
        shape = shape[0]
    return _shape_to_list(shape)


def _iter_layers(layer):
    yield layer
    for child in getattr(layer, "layers", []):
        yield from _iter_layers(child)


def _layer_class_names() -> list[str]:
    return [layer.__class__.__name__ for layer in _iter_layers(_model)]


_LAYER_CLASS_NAMES = _layer_class_names()
_HAS_INTERNAL_RESCALING = "Rescaling" in _LAYER_CLASS_NAMES
_HAS_INTERNAL_CENTER_CROP = "CenterCrop" in _LAYER_CLASS_NAMES
_INPUT_SHAPE = _model_input_shape()
_OUTPUT_SHAPE = _model_output_shape()
_INPUT_HEIGHT = int(_INPUT_SHAPE[1])
_INPUT_WIDTH = int(_INPUT_SHAPE[2])

logger.info("Loaded artifact model from %s", _MODEL_PATH)
logger.info("Model input shape: %s", _INPUT_SHAPE)
logger.info("Model output shape: %s", _OUTPUT_SHAPE)
logger.info("Loaded class_names order: %s", _class_names)
logger.info(
    "Internal preprocessing layers: rescaling=%s center_crop=%s",
    _HAS_INTERNAL_RESCALING,
    _HAS_INTERNAL_CENTER_CROP,
)


def is_model_ready() -> bool:
    return os.path.isfile(_MODEL_PATH)


def get_model_info() -> dict:
    return {
        "model_path": _MODEL_PATH,
        "model_filename": _MODEL_FILENAME,
        "model_input_shape": _INPUT_SHAPE,
        "model_output_shape": _OUTPUT_SHAPE,
        "number_of_classes": len(_class_names),
        "class_names": _class_names,
        "has_internal_rescaling": _HAS_INTERNAL_RESCALING,
        "has_internal_center_crop": _HAS_INTERNAL_CENTER_CROP,
        "layer_classes": _LAYER_CLASS_NAMES,
    }


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((_INPUT_WIDTH, _INPUT_HEIGHT))

    img_array = np.asarray(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    if not _HAS_INTERNAL_RESCALING:
        img_array = img_array / 255.0

    logger.info("Input image shape before prediction: %s", img_array.shape)
    logger.info(
        "Input image value range before prediction: min=%.4f max=%.4f",
        float(np.min(img_array)),
        float(np.max(img_array)),
    )
    return img_array


def _top_predictions(probabilities: np.ndarray, limit: int = 3) -> list[dict]:
    indexes = np.argsort(probabilities)[::-1][:limit]
    return [
        {
            "class_name": _class_names[int(index)],
            "confidence": round(float(probabilities[int(index)]), 4),
        }
        for index in indexes
    ]


def predict_image(image_bytes: bytes) -> dict:
    img_array = _preprocess_image(image_bytes)
    predictions = _model.predict(img_array, verbose=0)
    probabilities = np.asarray(predictions[0], dtype=np.float32)

    if len(probabilities) != len(_class_names):
        raise ValueError(
            f"Model output has {len(probabilities)} classes, but class_names.json has {len(_class_names)}."
        )

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    all_probabilities = {
        _class_names[index]: round(float(probability), 4)
        for index, probability in enumerate(probabilities)
    }
    top_predictions = _top_predictions(probabilities)

    logger.info("Prediction probabilities for all classes: %s", all_probabilities)
    logger.info("Top 3 predicted classes: %s", top_predictions)

    return {
        "class_name": _class_names[predicted_index],
        "confidence": round(confidence, 4),
        "top_predictions": top_predictions,
    }
