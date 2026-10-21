# Visual Object Sorting Robot — Computer Vision Module

Real-time recognition of **bottle, can, box and ball** from a webcam or a still image.
The system finds the object, draws a colour-coded box around it, and shows the class and confidence.
It is the vision component of our **Group Specialization 8 — Visual Object Sorting Robot**
(Computer Vision, BS Computer Science 4B).

```
Webcam frame ─► YOLO-World (finds up to 5 candidate boxes)
             ─► square crop + letterbox to 224×224
             ─► MobileNetV2 classifier (ball / bottle / box / can)
             ─► confidence filter ─► tracker (smoothing + voting)
             ─► box + label + confidence on screen
```

- **Detector:** YOLO-World `yolov8s-worldv2` (pretrained, prompts: bottle, can, box, ball). It only proposes *where* objects are.
- **Classifier:** MobileNetV2 trained by the group with transfer learning. It decides *what* each object is.
- **Tracker:** keeps the box and label steady across frames and marks a result `CONFIRMED` after 5 agreeing checks.

## Results at a glance

| Metric | Value |
| --- | --- |
| Validation accuracy | **95.5%** (192 of 201 images) |
| Average precision / recall / F1 | 96.0% / 96.1% / 96.0% |
| Live recognition | All 4 classes CONFIRMED at **97–99%** confidence |
| Unseen objects (toothbrush, strainer, open hand) | Correctly ignored ("No object") |
| Known failure | Very dark lighting: the object is not detected |

Full midterm report: [`docs/`](docs/) · Details in [Results](#results) below.

---

## Project structure

```
Object-Sorting/
├── collect_data.py        # capture your own training photos from the webcam
├── train.py               # train the MobileNetV2 classifier
├── evaluate.py            # accuracy, precision, recall, F1, confusion matrix
├── classify_live.py       # run recognition on the webcam or an image
├── detect.py              # object finder (YOLO-World / background subtraction), cropping, drawing
├── model.keras            # trained classifier (4 classes)
├── labels.txt             # class order: ball, bottle, box, can
├── yolov8s-worldv2.pt     # YOLO-World detector weights
├── requirements.txt
├── docs/
│   └── Team_Bulastog-Object-sorting.pdf   # midterm project report
└── dataset/
    └── dataset.rar        # 1,011 images in 4 class folders (extract before training)
```

> If `dataset.rar` is not in the repository, download it from the **Releases** page
> (it is larger than GitHub's 25 MB web-upload limit).

## Requirements

- Python **3.9 – 3.12** (TensorFlow may not support newer versions yet)
- A webcam (tested at 640×480)
- Runs on CPU; a GPU is optional

```
pip install -r requirements.txt
```

| Package | Version |
| --- | --- |
| tensorflow | ≥ 2.13 |
| opencv-python | ≥ 4.8 |
| ultralytics | ≥ 8.3 |
| numpy | ≥ 1.24 |
| pillow | ≥ 10.0 |

## Quick start (use the trained model)

The repository already includes `model.keras` and `labels.txt`, so you can run it right away:

```
python classify_live.py                  # live webcam
python classify_live.py photo.jpg        # single image -> saves photo_labeled.jpg
```

Press **q** to quit.

---

## 1. Dataset

Extract `dataset/dataset.rar` (7-Zip or WinRAR → *Extract Here*) so the images sit directly in one
folder per class. If extracting creates an extra `dataset/dataset/` level, move the four class folders
up one level, or pass `--data dataset/dataset` to the scripts.

```
dataset/
├── ball/     297 images  (9 ball types: baseball, basketball, billiard, bowling,
│                          cricket, football, golf, tennis, volleyball)
├── bottle/   300 images  (incl. 100 crushed bottles)
├── box/      260 images
└── can/      154 images
              -----
              1,011 images
```

### Adding your own photos (recommended)

Photos from the real sorting setup make a big difference.

```
python collect_data.py
```

| Key | Saves to |
| --- | --- |
| `1` | bottle |
| `2` | can |
| `3` | box |
| `4` | ball |
| `5` | other |
| `q` | quit |

Each photo is saved as a clean square crop around the outlined object (no drawings on it).

### The `other` class

The current model has **no `other` class**, so it must call every crop one of the four objects —
a hand or arm can be labelled as a bottle. To fix this, collect **30+ photos each** of your
hand/arm in view, the empty table and random clutter with key `5`, then retrain.
With `other` present, these scenes are ignored and the screen shows "No object".

## 2. Train

```
python train.py                                  # reads ./dataset
python train.py --data my_folder --epochs 15     # custom folder / epochs
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | `dataset` | Dataset folder (one subfolder per class) |
| `--img` | 224 | Input image size |
| `--batch` | 16 | Batch size |
| `--epochs` | 12 | Max epochs, phase 1 (head only) |
| `--fine-epochs` | 10 | Max epochs, phase 2 (fine-tuning) |
| `--no-pretrained` | off | Skip ImageNet weights (offline testing only) |

What the script does:

- **Split:** stratified 80% training / 20% validation per class, seed 42.
- **Cleaning:** skips unsupported formats (`.webp`, `.tif`) and drops corrupt images.
- **Resize:** letterbox to 224×224 — padded, never squashed.
- **Class balance:** class weights so the smaller `can` class counts more.
- **Augmentation:** horizontal flip, rotation ±36°, zoom ±25%, shift ±15%, brightness ±30%, contrast ±30%, Gaussian noise.
- **Phase 1:** frozen MobileNetV2 backbone, train the new head — Adam, lr 1e-3.
- **Phase 2:** unfreeze the last 30 backbone layers — Adam, lr 1e-5.
- **Callbacks:** early stopping (val accuracy, patience 3) and learning-rate halving (val loss, patience 2).

At the end it prints the **validation accuracy** and a **confusion matrix with recall per class**,
and saves `model.keras` + `labels.txt`. The first run needs internet to download the ImageNet weights.

## 3. Evaluate

```
pip install matplotlib
python evaluate.py                       # or: python evaluate.py --data dataset/dataset
```

Rebuilds the same 80/20 validation split as `train.py` (seed 42), runs `model.keras` on it, and prints
accuracy plus precision, recall and F1 per class. It also saves `confusion_matrix.png` and
`sample_predictions.png` (green = correct, red = wrong).

## 4. Run

```
python classify_live.py                  # webcam
python classify_live.py photo.jpg        # image -> photo_labeled.jpg
```

### On-screen output

| Element | Meaning |
| --- | --- |
| Coloured box | bottle = blue, can = yellow, box = green, ball = red |
| `bottle 94%` | predicted class and confidence |
| `CONFIRMED` + thick box | the last 5 checks all agreed |
| `No object` | nothing passed the confidence filter |
| `cam 30 fps \| det 5.2/s` | video frame rate \| detection checks per second |

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode` | `yolo` | Object finder: `yolo`, `auto` (YOLO, else background), `background`, `none` (whole frame) |
| `--camera` | 0 | Camera index (try `1` if the camera does not open) |
| `--conf` | 0.80 | Confidence needed to start tracking a **new** object |
| `--keep-conf` | 0.50 | Confidence needed to **keep** tracking an object |
| `--hold` | 0.8 | Seconds the box stays after the object was last seen |
| `--det-conf` | 0.10 | Minimum YOLO-World confidence for candidate boxes |
| `--imgsz` | 416 | Detector input size (smaller = faster, e.g. 320) |
| `--no-refine` | off | Skip the GrabCut outline (faster) |

**Background mode:** with `--mode background`, clear the sorting area and press **b** to save the
empty scene; objects are then found as differences from it.

---

## Results

### Offline validation (201 images, `evaluate.py`)

| Class | Precision | Recall | F1-score | Images |
| --- | --- | --- | --- | --- |
| Ball | 89.2% | 98.3% | 93.5% | 59 |
| Bottle | 98.2% | 90.0% | 93.9% | 60 |
| Box | 100.0% | 96.2% | 98.0% | 52 |
| Can | 96.8% | 100.0% | 98.4% | 30 |
| **Average** | **96.0%** | **96.1%** | **96.0%** | **201** |

Confusion matrix (rows = true class, columns = predicted):

| True \ Predicted | Ball | Bottle | Box | Can |
| --- | --- | --- | --- | --- |
| Ball | **58** | 1 | 0 | 0 |
| Bottle | 5 | **54** | 0 | 1 |
| Box | 2 | 0 | **50** | 0 |
| Can | 0 | 0 | 0 | **30** |

7 of the 9 errors were predicted as **ball**; the most common mistake is a clear or crushed bottle
read as a ball (5 of 60).

### Live webcam tests (`classify_live.py`)

| Test | Result |
| --- | --- |
| Red product box | box 99%, CONFIRMED |
| Clear plastic water bottle | bottle 98%, CONFIRMED |
| Canned tuna | can 97%, CONFIRMED |
| Small soccer-style ball | ball 99%, CONFIRMED |
| Empty scene | No object ✔ |
| Toothbrush (not a trained class) | No object ✔ |
| Round pink strainer (looks like a ball) | No object ✔ |
| Open hand only | No object ✔ |
| Object in a very dark room | No object ✘ (too dark to detect) |

Camera ran at 15 fps; detection ran at 4–13 checks per second on a laptop CPU
(Intel Core vPro i5, 16 GB RAM).

---

## Troubleshooting

**Box flickers or disappears**
Try a steadier setting: `--hold 1.5 --keep-conf 0.4`.
If an object keeps getting rejected, add more photos of it from your own setup and retrain.

**Low FPS / slow detection**
The video loop runs separately from detection, so the video stays smooth while the box updates a
few times per second. To raise `det`: use `--imgsz 320` and `--no-refine`.
TensorFlow on native Windows runs on the CPU only; a modest laptop gets about 3–10 detection
updates per second.

**`Dataset classes ['dataset'] do not match labels.txt`**
The dataset was extracted one folder too deep. Use `--data dataset/dataset` or move the class folders up.

**It keeps detecting my arm or hand**
1. Add an `other` class (see above) and retrain.
2. Raise the threshold: `--conf 0.9`.
3. Keep your arm out of the frame while saving photos for the object classes.

**"Could not open camera"**
Use `--camera 1` (or 2) to select another device.

**"YOLO-World not available"**
Run `pip install ultralytics` and make sure `yolov8s-worldv2.pt` is in the project folder,
or run with `--mode auto` to fall back to background subtraction.

---

## Known limitations

- Only one object (the most confident) is reported per frame.
- No separate test set yet — accuracy is measured on the validation set.
- No `other` class in the current model (in testing, the 0.80 threshold still kept hands from being labelled).
- Needs adequate lighting; objects are not detected in a very dark room.
- Most training images come from public datasets, not our own camera.
- The robot and sorting mechanism are planned for the Final project.

## Group

Villamer, Gracielle Ann Z. · Pimentel, Deiven Joseph I. · Romance, Jeric E. ·
Ibarrientos, Jenny J. · Namia, Erica Denese L. · Biares, Francisc Louise Q.

BS Computer Science 4B — Computer Vision · Instructor: Elena Clarexe Gulane

## References

- Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
- Cheng, T., et al. (2024). YOLO-World: Real-time open-vocabulary object detection. *CVPR*.
- Jocher, G., Qiu, J., & Chaurasia, A. (2023). *Ultralytics YOLO* (Version 8). https://github.com/ultralytics/ultralytics
- Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut. *ACM Transactions on Graphics, 23*(3).
