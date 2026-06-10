"""
Egyptian Artifact Recognition — Training Script (Clean Rewrite)
================================================================
Simplified to avoid pickle/serialization issues.
Augmentation & normalization are applied via dataset .map() instead
of embedding layers inside the model, so no complex objects need
to be serialized between epochs.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ── Config ──────────────────────────────────────────
TRAIN_DIR   = "train"
MODEL_PATH  = "model/model.h5"
NAMES_PATH  = "model/class_names.json"
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 100

# ── Load datasets ───────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes: {class_names}")

with open(NAMES_PATH, "w") as f:
    json.dump(class_names, f)
print(f"📄 Class names saved to {NAMES_PATH}")

# ── Class weights ────────────────────────────────────
labels = []
for _, y in train_ds:
    labels.extend(np.argmax(y.numpy(), axis=1))

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = {int(i): float(w) for i, w in enumerate(weights)}
print(f"⚖️  Class weights: {class_weight_dict}")

# ── Preprocessing & Augmentation ────────────────────
normalization = tf.keras.layers.Rescaling(1./255)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="augmentation")

train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x), training=True), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# ── Build model ──────────────────────────────────────
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all then unfreeze last 30
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"🔓 Last 30 layers unfrozen out of {len(base_model.layers)}")

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dense(256, activation="relu")(x)
x       = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model   = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    ),
]

# ── Train ────────────────────────────────────────────
print("🚀 Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("✅ Training complete!")
print(f"   Model saved to: {MODEL_PATH}")
