# Visual Object Sorting Robot – Midterm: 4-Class Recognition

Recognizes **bottle, can, box, ball** from a camera, outlines the object and labels it.
Classifier: MobileNetV2 transfer learning. Works with most public datasets and cluttered backgrounds.

## Setup
```
pip install -r requirements.txt
```
Use Python 3.9–3.12 (TensorFlow may not support newer versions yet).


## 1. Get the data
**Add photos from your own robot camera.**
```
python collect_data.py
```
Press `1` bottle, `2` can, `3` box, `4` ball when the object is outlined. photos from your
real setup makes a big difference.

**Also collect an `other` class (key `5`)**: 30+ photos each of your arm/hand in view, the empty table,
and random clutter. Without it the model must call everything one of the four classes, so it will
label your arm as a bottle. With it, arms/hands/empty scenes are ignored ("No object").
`prepare_dataset.py` also fills `other` automatically from classes like person/hand/arm/background
(add `--all-unmapped-as-other` to include every other class such as cat, chair, ...).

## 2. Train
```
python train.py              # reads ./dataset
python train.py --data my_folder --epochs 15
```
Handles corrupt images, unbalanced classes (class weights), any image size/aspect ratio (padded, never
squashed), and uses heavy augmentation (flip, rotation ±36°, zoom, shifts, brightness, contrast, noise).
Prints validation accuracy and a confusion matrix, and writes `model.keras` + `labels.txt`.
Needs internet the first time to download the ImageNet weights.

## 3. Run
```
python classify_live.py                  # webcam, object outlined + labeled
python classify_live.py photo.jpg        # saves photo_labeled.jpg
```

The finder proposes several candidate boxes; each is classified and the most confident real object wins.
Boxes classed as `other` or below 80% confidence are ignored. The box thickens with `CONFIRMED` after
5 matching frames. Tune with `--conf 0.9` (stricter) and `--det-conf` (detector sensitivity).

**Outline flickering / disappearing?** `classify_live.py` has a built-in tracker: the outline stays through
brief misses (`--hold`, default 0.8 s, automatically longer on slow computers), an object already being
tracked only needs `--keep-conf` (default 0.5) to stay while a new one needs `--conf` (0.8), the box
and outline are smoothed, and the label is a majority vote that only changes when another class clearly
outweighs it. For a steadier view try `--hold 1.5 --keep-conf 0.4`. If it flickers because the object
keeps being classed `other`, add more photos of that object (in your own setup) and retrain.

**Low FPS?** The video loop runs separately from detection, so the video stays smooth (`cam` fps)
while the box updates a few times per second (`det`). To raise `det`: use `--imgsz 320` (smaller
detector input), `--no-refine` (skip the GrabCut outline). The camera is opened as 640x480 MJPG, which is much faster than the default on
most webcams. TensorFlow on native Windows runs on the CPU only, so a modest laptop gets roughly
3-10 detection updates per second in yolo mode.

**It keeps detecting my arm?** (1) Train with an `other` class containing your arm/hand/empty table
(see above) and retrain; (2) raise `--conf` to 0.9; (3) keep your arm out of the frame while saving
photos for the object classes.