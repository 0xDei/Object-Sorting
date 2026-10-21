import cv2
import numpy as np

DIFF_THRESHOLD = 30       # background mode: how different a pixel must be (0-255)
MIN_AREA_FRACTION = 0.01  # background mode: ignore blobs under 1% of the frame
YOLO_WEIGHTS = "yolov8s-worldv2.pt"
YOLO_PROMPTS = ["bottle", "can", "box", "ball"]
YOLO_CONF = 0.10          # candidate boxes only; the classifier makes the final call
YOLO_IMGSZ = 416          # detector input size: smaller = faster (640 is the slow default)

COLORS = {               # BGR outline colors per class
    "bottle": (255, 140, 0),
    "can": (0, 200, 255),
    "box": (0, 200, 0),
    "ball": (60, 60, 255),
    "unknown": (160, 160, 160),
}


def _prep(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


def refine_outline(frame, bbox, max_side=220):
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    pad = int(0.1 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    scale = min(1.0, max_side / max(roi.shape[:2]))
    small = cv2.resize(roi, None, fx=scale, fy=scale) if scale < 1 else roi.copy()
    rect = (int((x - x0) * scale), int((y - y0) * scale),
            max(2, int(w * scale)), max(2, int(h * scale)))
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * rect[2] * rect[3]:   # grabcut failed; keep plain box
        return None
    return (c.astype(np.float32) / scale + np.array([x0, y0], np.float32)).astype(np.int32)


class Localizer:
    def __init__(self, mode="yolo", refine=True, det_conf=YOLO_CONF, imgsz=YOLO_IMGSZ):
        self.refine = refine
        self.det_conf = det_conf
        self.imgsz = imgsz
        self.background = None
        self.yolo = None
        self.mode = mode
        if mode in ("auto", "yolo"):
            self.yolo = self._load_yolo(required=(mode == "yolo"))
            if mode == "auto":
                self.mode = "yolo" if self.yolo is not None else "background"
        print(f"Object finder: {self.mode}")

    @staticmethod
    def _load_yolo(required):
        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(YOLO_WEIGHTS)
            model.set_classes(YOLO_PROMPTS)
            return model
        except Exception as e:
            msg = f"YOLO-World not available ({type(e).__name__}: {e})"
            if required:
                raise SystemExit(msg + "\nInstall with: pip install ultralytics")
            print(msg + "\nFalling back to background subtraction "
                  "(pip install ultralytics for cluttered backgrounds).")
            return None

    @property
    def needs_background(self):
        return self.mode == "background"

    def set_background(self, frame):
        self.background = _prep(frame)

    def candidates(self, frame, max_n=5):
        H, W = frame.shape[:2]
        if self.mode == "none":
            return [(None, (0, 0, W, H))]

        if self.mode == "yolo":
            res = self.yolo.predict(frame, conf=self.det_conf, imgsz=self.imgsz,
                                    verbose=False)[0]
            confs = res.boxes.conf.tolist() if len(res.boxes) else []
            order = sorted(range(len(confs)), key=lambda i: -confs[i])
            out = []
            for i in order[:max_n]:
                x1, y1, x2, y2 = [float(v) for v in res.boxes.xyxy[i].tolist()]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W, int(x2)), min(H, int(y2))
                w, h = x2 - x1, y2 - y1
                if w < 4 or h < 4 or w * h > 0.9 * W * H:   # ignore slivers and whole-frame boxes
                    continue
                out.append((None, (x1, y1, w, h)))
            return out

        # background subtraction: the changed blobs, largest first
        if self.background is None:
            return []
        diff = cv2.absdiff(self.background, _prep(frame))
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA_FRACTION * H * W]
        contours.sort(key=cv2.contourArea, reverse=True)
        return [(c, cv2.boundingRect(c)) for c in contours[:max_n]]

    def outline(self, frame, bbox):
        if self.mode == "yolo" and self.refine:
            return refine_outline(frame, bbox)
        return None

    def find(self, frame):
        cands = self.candidates(frame, 1)
        if not cands:
            return None
        contour, bbox = cands[0]
        if contour is None:
            contour = self.outline(frame, bbox)
        return contour, bbox


def crop_square(frame, bbox, pad=0.15):
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    side = int(max(w, h) * (1 + 2 * pad))
    side = max(32, min(side, H, W))
    cx, cy = x + w // 2, y + h // 2
    x0 = int(np.clip(cx - side // 2, 0, W - side))
    y0 = int(np.clip(cy - side // 2, 0, H - side))
    return frame[y0:y0 + side, x0:x0 + side]


def draw_detection(frame, contour, bbox, text, label="unknown", confirmed=False):
    color = COLORS.get(label, COLORS["unknown"])
    x, y, w, h = bbox
    # if contour is not None:
    #     cv2.drawContours(frame, [contour], -1, color, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3 if confirmed else 1)

    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = y - 8 if y - th - base - 8 > 0 else y + h + th + 8
    cv2.rectangle(frame, (x, ty - th - base), (x + tw + 8, ty + base), color, -1)
    cv2.putText(frame, text, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)