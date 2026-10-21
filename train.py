"""
Step 2: Train the classifier (bottle, can, box, ball).

Transfer learning from MobileNetV2 (pretrained on ImageNet). Built to cope with
messy data: corrupt images are skipped, unbalanced classes are re-weighted,
photos have any size/aspect ratio (letterboxed, never squashed), and heavy
augmentation (shifts, zoom, rotation, lighting, noise) helps with cluttered
backgrounds.

Usage:
    python train.py                       # uses ./dataset
    python train.py --data my_folder --epochs 15

Dataset layout: <data>/<class_name>/**/*.jpg  (use prepare_dataset.py to
convert YOLO/COCO/VOC/zip datasets into this layout).

Output:
    model.keras   the trained model
    labels.txt    class names in the order the model outputs them
"""
import argparse
import os
import random
import sys
from collections import Counter

import numpy as np
import tensorflow as tf

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}      # what TensorFlow can decode
CONVERTIBLE = {".webp", ".tif", ".tiff", ".jfif"}           # prepare_dataset.py handles these

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--img", type=int, default=224)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs", type=int, default=12, help="max epochs for the first phase")
ap.add_argument("--fine-epochs", type=int, default=10, help="max epochs for fine-tuning")
ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline testing only)")
args = ap.parse_args()

IMG = (args.img, args.img)
AUTOTUNE = tf.data.AUTOTUNE
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------------ collect files
items, skipped = [], 0
if not os.path.isdir(args.data):
    sys.exit(f"Dataset folder '{args.data}' not found.")
for root, _, files in os.walk(args.data):
    rel = os.path.relpath(root, args.data)
    if rel == ".":
        continue
    cls = rel.split(os.sep)[0]            # class = first folder under the dataset root
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED:
            items.append((os.path.join(root, f), cls))
        elif ext in CONVERTIBLE:
            skipped += 1
if skipped:
    print(f"Skipped {skipped} .webp/.tif files. Run prepare_dataset.py to convert them.")

class_names = sorted({c for _, c in items})
if len(class_names) < 2:
    sys.exit("Need at least 2 class folders with images inside the dataset folder.")
counts = Counter(c for _, c in items)
print("Images per class:", dict(counts))
too_few = [c for c in class_names if counts[c] < 5]
if too_few:
    sys.exit(f"These classes have fewer than 5 images: {too_few}")
cidx = {c: i for i, c in enumerate(class_names)}

# stratified 80/20 split
train_items, val_items = [], []
for c in class_names:
    group = [it for it in items if it[1] == c]
    random.shuffle(group)
    n_val = max(1, int(0.2 * len(group)))
    val_items += group[:n_val]
    train_items += group[n_val:]
print(f"Train: {len(train_items)}  Validation: {len(val_items)}")

with open("labels.txt", "w") as f:
    f.write("\n".join(class_names))


# ------------------------------------------------------------------ input pipeline
def load(path, label):
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize_with_pad(img, IMG[0], IMG[1])     # keeps aspect ratio
    return tf.cast(img, tf.float32), label


def make_ds(its, training):
    paths = [p for p, _ in its]
    labels = [cidx[c] for _, c in its]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(its), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).ignore_errors()   # drops corrupt images
    return ds.batch(args.batch).prefetch(AUTOTUNE)


train_ds = make_ds(train_items, True)
val_ds = make_ds(val_items, False)

# balance classes
total = len(train_items)
tc = Counter(cidx[c] for _, c in train_items)
class_weight = {i: total / (len(class_names) * tc[i]) for i in range(len(class_names))}

# ------------------------------------------------------------------ model
L = tf.keras.layers
augment = tf.keras.Sequential([
    L.RandomFlip("horizontal"),
    L.RandomRotation(0.1),
    L.RandomZoom(0.25),
    L.RandomTranslation(0.15, 0.15),
    L.RandomBrightness(0.3),
    L.RandomContrast(0.3)
])

try:
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG + (3,), include_top=False,
        weights=None if args.no_pretrained else "imagenet")
except Exception as e:
    print(f"\nWARNING: could not load ImageNet weights ({e}).\n"
          "Training from scratch will be MUCH worse. Check your internet connection.\n")
    base = tf.keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights=None)
base.trainable = False

inputs = tf.keras.Input(shape=IMG + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = L.GaussianNoise(0.05)(x)
x = base(x, training=False)
x = L.GlobalAveragePooling2D()(x)
x = L.Dropout(0.3)(x)
outputs = L.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]


model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
          class_weight=class_weight, callbacks=callbacks())

# fine-tune the top of the backbone
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=args.fine_epochs,
          class_weight=class_weight, callbacks=callbacks())

# ------------------------------------------------------------------ report
y_true, y_pred = [], []
for xb, yb in val_ds:
    y_true += yb.numpy().tolist()
    y_pred += np.argmax(model.predict(xb, verbose=0), axis=1).tolist()
y_true, y_pred = np.array(y_true), np.array(y_pred)
print(f"\nValidation accuracy: {(y_true == y_pred).mean():.1%}  ({len(y_true)} images)")

n = len(class_names)
cm = np.zeros((n, n), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
w = max(len(c) for c in class_names) + 2
print("Confusion matrix (rows = true, columns = predicted):")
print(" " * w + "".join(f"{c[:8]:>9}" for c in class_names))
for i, c in enumerate(class_names):
    recall = cm[i, i] / max(1, cm[i].sum())
    print(f"{c:<{w}}" + "".join(f"{v:>9}" for v in cm[i]) + f"   recall {recall:.0%}")

model.save("model.keras")
print("Saved model.keras and labels.txt")
