import argparse
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

import detect

MODEL_PATH = "model.keras"
LABELS_PATH = "labels.txt"
IMG_SIZE = 224
CONF_THRESHOLD = 0.80    # needed to start tracking a NEW object
KEEP_THRESHOLD = 0.50    # enough to keep tracking an object we already have
HOLD_SECONDS = 0.80      # keep showing the object this long after it is last seen
STABLE_FRAMES = 5        # last N checks all agree -> CONFIRMED
VOTE_WINDOW = 7          # label = majority vote over the last N checks
BOX_ALPHA_DET = 0.6      # how strongly a new detection pulls the tracked box
BOX_ALPHA_SHOW = 0.4     # per-video-frame easing of the displayed box
IGNORE = {"unknown", "other"}

model = tf.keras.models.load_model(MODEL_PATH)
labels = open(LABELS_PATH).read().split()
if "other" not in labels:
    print("Note: this model has no 'other' class, so it cannot tell an arm/hand from an object.\n"
          "Add photos of your arm/hand/empty table to dataset/other/ and retrain.")


@tf.function(reduce_retracing=True)
def _infer(x):
    return model(x, training=False)


_infer(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))   # warm-up so the first frame isn't slow


def letterbox(img, size=IMG_SIZE):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    resized = cv2.resize(img, (nw, nh),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def classify_batch(crops, apply_threshold=True):
    if not crops:
        return []
    batch = np.stack([cv2.cvtColor(letterbox(c), cv2.COLOR_BGR2RGB) for c in crops])
    probs = _infer(tf.constant(batch.astype("float32"))).numpy()
    out = []
    for p in probs:
        i = int(np.argmax(p))
        conf = float(p[i])
        if apply_threshold and conf < CONF_THRESHOLD:
            out.append(("unknown", conf))
        else:
            out.append((labels[i], conf))
    return out


def classify(image_bgr):
    return classify_batch([image_bgr])[0]


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def best_detection(frame, localizer, track_bbox=None):
    cands = localizer.candidates(frame)
    if not cands:
        return None
    results = classify_batch([detect.crop_square(frame, bbox) for _, bbox in cands],
                             apply_threshold=False)
    best = None
    for (contour, bbox), (label, conf) in zip(cands, results):
        if label in IGNORE:
            continue
        on_track = track_bbox is not None and iou(bbox, track_bbox) > 0.3
        if conf < (KEEP_THRESHOLD if on_track else CONF_THRESHOLD):
            continue
        score = conf + (0.15 if on_track else 0.0)
        if best is None or score > best[0]:
            best = (score, label, conf, contour, bbox)
    if best is None:
        return None
    _, label, conf, contour, bbox = best
    if contour is None:
        contour = localizer.outline(frame, bbox)
    return label, conf, contour, bbox


def _resample(poly, n=48):
    pts = poly.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None
    k = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    pts = np.roll(pts, -k, axis=0)
    closed = np.vstack([pts, pts[:1]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    if cum[-1] < 1e-3:
        return None
    t = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])], axis=1)


def _relative_hull(contour, bbox):
    pts = _resample(cv2.convexHull(contour))
    if pts is None:
        return None
    x, y, w, h = bbox
    return (pts - np.array([x, y], np.float32)) / np.array([max(w, 1), max(h, 1)], np.float32)


class Tracker:
    def __init__(self, hold=HOLD_SECONDS):
        self.hold = hold
        self.dt = 0.3            # running average of seconds between detection checks
        self.prev = None
        self.lock = threading.Lock()
        self._clear()

    def _hold(self):
        return max(self.hold, 3.0 * self.dt)

    def _clear(self):
        self.votes = deque(maxlen=VOTE_WINDOW)
        self.target = None       # smoothed box from detections
        self.shown = None        # box eased toward target every video frame
        self.rel = None          # outline relative to the box
        self.shown_label = None  # label currently displayed (only changes on a clear majority)
        self.last_seen = 0.0

    def reset(self):
        with self.lock:
            self._clear()

    def target_bbox(self):
        with self.lock:
            return None if self.target is None else tuple(float(v) for v in self.target)

    def update(self, found):
        now = time.time()
        with self.lock:
            if self.prev is not None:
                self.dt = 0.8 * self.dt + 0.2 * min(now - self.prev, 2.0)
            self.prev = now
            if found is None:
                if self.target is not None and now - self.last_seen > self._hold():
                    self._clear()
                return
            label, conf, contour, bbox = found
            new = np.array(bbox, np.float32)
            if self.target is not None and iou(self.target, new) < 0.2:
                self._clear()                      # a different object elsewhere
            if self.target is None:
                self.target, self.shown = new.copy(), new.copy()
            else:
                self.target = BOX_ALPHA_DET * new + (1 - BOX_ALPHA_DET) * self.target
            self.votes.append((label, conf))
            self.last_seen = now
            if contour is not None:
                rel = _relative_hull(contour, bbox)
                if rel is not None:
                    self.rel = rel if self.rel is None else 0.6 * rel + 0.4 * self.rel

    def snapshot(self):
        now = time.time()
        with self.lock:
            if self.target is None:
                return None
            if now - self.last_seen > self._hold():
                self._clear()
                return None
            self.shown = self.shown + BOX_ALPHA_SHOW * (self.target - self.shown)
            sums = {}
            for lab, c in self.votes:
                sums[lab] = sums.get(lab, 0.0) + c
            top = max(sums, key=sums.get)
            cur = self.shown_label
            # switch label only if the challenger clearly outweighs the current one
            if cur is None or cur not in sums or (top != cur and sums[top] > 1.3 * sums[cur]):
                cur = top
            self.shown_label = label = cur
            conf = float(np.mean([c for lab, c in self.votes if lab == label]))
            recent = [lab for lab, _ in list(self.votes)[-STABLE_FRAMES:]]
            confirmed = len(recent) == STABLE_FRAMES and all(lab == label for lab in recent)
            x, y, w, h = [float(v) for v in self.shown]
            contour = None
            if self.rel is not None:
                pts = self.rel * np.array([w, h], np.float32) + np.array([x, y], np.float32)
                contour = pts.astype(np.int32).reshape(-1, 1, 2)
            return dict(label=label, conf=conf,
                        bbox=(int(x), int(y), max(1, int(w)), max(1, int(h))),
                        contour=contour, confirmed=confirmed)


class Worker(threading.Thread):
    def __init__(self, localizer, tracker):
        super().__init__(daemon=True)
        self.localizer = localizer
        self.tracker = tracker
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.frame = None
        self.fps = 0.0
        self.running = True

    def submit(self, frame):
        with self.lock:
            self.frame = frame
        self.wake.set()

    def run(self):
        last = time.time()
        while self.running:
            self.wake.wait(0.5)
            self.wake.clear()
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                continue
            if self.localizer.needs_background and self.localizer.background is None:
                found = None
            else:
                found = best_detection(frame, self.localizer, self.tracker.target_bbox())
            self.tracker.update(found)
            now = time.time()
            self.fps = 0.8 * self.fps + 0.2 / max(now - last, 1e-6)
            last = now


def run_image(path, localizer):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read {path}")
    found = best_detection(img, localizer)
    if found is None:
        print("No object found (nothing detected, or it was classed as other/unsure).")
        return
    label, conf, contour, bbox = found
    print(f"{label} ({conf:.0%})")
    detect.draw_detection(img, contour, bbox, f"{label} {conf:.0%}", label, True)
    out = os.path.splitext(path)[0] + "_labeled.jpg"
    cv2.imwrite(out, img)
    print(f"Saved {out}")


def open_camera(index):
    cap = None
    if os.name == "nt":                         # DirectShow is faster to open on Windows
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    global CONF_THRESHOLD, KEEP_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--mode", default="yolo", choices=["yolo", "auto", "background", "none"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                    help="classifier confidence needed to start tracking a new object (0-1)")
    ap.add_argument("--keep-conf", type=float, default=KEEP_THRESHOLD,
                    help="confidence needed to KEEP tracking an object (lower = steadier)")
    ap.add_argument("--hold", type=float, default=HOLD_SECONDS,
                    help="seconds to keep the outline after the object was last seen")
    ap.add_argument("--det-conf", type=float, default=detect.YOLO_CONF,
                    help="minimum detector confidence for candidate boxes (yolo mode)")
    ap.add_argument("--imgsz", type=int, default=detect.YOLO_IMGSZ,
                    help="detector input size; smaller is faster (e.g. 320)")
    ap.add_argument("--no-refine", action="store_true", help="box only, skip the GrabCut outline (faster)")
    args = ap.parse_args()
    CONF_THRESHOLD = args.conf
    KEEP_THRESHOLD = min(args.keep_conf, args.conf)

    mode = args.mode
    if args.image and mode in ("auto", "background"):
        mode = "auto"   # no background photo for a single image
    localizer = detect.Localizer(mode, refine=not args.no_refine,
                                 det_conf=args.det_conf, imgsz=args.imgsz)
    if args.image and localizer.mode == "background":
        localizer = detect.Localizer("none")
    if args.image:
        return run_image(args.image, localizer)

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit("Could not open camera. Try --camera 1.")

    if localizer.needs_background:
        print("Clear the sorting area and press 'b' to save the background. 'q' quits.")

    tracker = Tracker(hold=args.hold)
    worker = Worker(localizer, tracker)
    worker.start()
    cam_fps, last = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            localizer.set_background(frame)
            tracker.reset()
            print("Background saved.")

        worker.submit(frame)                       # newest frame; detection catches up on its own
        snap = tracker.snapshot()

        view = frame.copy()
        if localizer.needs_background and localizer.background is None:
            cv2.putText(view, "Empty the area, then press B", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        elif snap is None:
            cv2.putText(view, "No object", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        else:
            text = f"{snap['label']} {snap['conf']:.0%}" + (" CONFIRMED" if snap["confirmed"] else "")
            detect.draw_detection(view, snap["contour"], snap["bbox"], text,
                                  snap["label"], snap["confirmed"])

        now = time.time()
        cam_fps = 0.9 * cam_fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(view, f"cam {cam_fps:.0f} fps | det {worker.fps:.1f}/s",
                    (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.imshow("classify_live", view)

    worker.running = False
    worker.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()