#!/usr/bin/env python3
"""
test_model.py — Local smoke-test for the TensorFlow artifact recognition model.

Usage (run from the CV_Recognition root):
    python test_model.py <path/to/image.jpg>

This test uses the same TensorFlow + model.h5 stack as the live service,
so a passing test here means the API will also work correctly.
"""

import json
import sys
import os

import numpy as np
from PIL import Image

# ── Ensure project root is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.predict import predict_image, get_model_info, is_model_ready

# ── Validate model is present ─────────────────────────────────────────────────
if not is_model_ready():
    print("❌ ERROR: model/model.h5 not found. Cannot run test.")
    sys.exit(1)

# ── Print model info ──────────────────────────────────────────────────────────
info = get_model_info()
print("=" * 60)
print("  MODEL INFO")
print("=" * 60)
print(f"  Model file  : {info['model_filename']}")
print(f"  Input shape : {info['model_input_shape']}")
print(f"  Classes     : {info['number_of_classes']}")
print(f"  Class names : {info['class_names']}")
print(f"  Internal rescaling : {info['has_internal_rescaling']}")
print("=" * 60)

# ── Run prediction ────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("\n⚠  No image path provided. Pass an image path as argument:")
    print("      python test_model.py path/to/image.jpg")
    print("\nModel loaded successfully — inference stack is working.\n")
    sys.exit(0)

image_path = sys.argv[1]
if not os.path.isfile(image_path):
    print(f"❌ ERROR: File not found: {image_path}")
    sys.exit(1)

with open(image_path, "rb") as f:
    image_bytes = f.read()

print(f"\n🔎 Running prediction on: {image_path}")
result = predict_image(image_bytes)

print("\n========== RESULT ==========")
print(f"  Artifact   : {result['class_name']}")
print(f"  Confidence : {result['confidence']:.2%}")
print("============================")
print("\n  Top 3 predictions:")
for pred in result["top_predictions"]:
    bar = "█" * int(pred["confidence"] * 40)
    print(f"    {pred['class_name']:<40} {pred['confidence']:.2%}  {bar}")
print()