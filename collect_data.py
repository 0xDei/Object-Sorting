import argparse
import os
import time

import cv2

import detect

CLASSES = ["bottle", "can", "box", "ball", "other"]
DATA_DIR = "dataset"

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="yolo", choices=["yolo", "auto", "background", "none"])
ap.add_argument("--camera", type=int, default=0)
args = ap.parse_args()

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)


def count(c):
    return len(os.listdir(os.path.join(DATA_DIR, c)))


localizer = detect.Localizer(args.mode)
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("Could not open camera. Try --camera 1.")

message, message_until = "", 0
print("Press 1-4 to save. 'q' quits." +
      (" Empty the area and press 'b' first." if localizer.needs_background else ""))

while True:
    ok, frame = cap.read()
    if not ok:
        break

    found = localizer.find(frame)
    view = frame.copy()
    if found is not None and found[1] != (0, 0, frame.shape[1], frame.shape[0]):
        detect.draw_detection(view, found[0], found[1], "object", "unknown")

    y = 25
    for i, c in enumerate(CLASSES, start=1):
        cv2.putText(view, f"[{i}] {c}: {count(c)}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 28
    if localizer.needs_background and localizer.background is None:
        cv2.putText(view, "Empty the area, press B", (10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    if time.time() < message_until:
        cv2.putText(view, message, (10, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("collect_data", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("b"):
        localizer.set_background(frame)
        message, message_until = "Background saved", time.time() + 1.5
    elif ord("1") <= key <= ord("5"):
        c = CLASSES[key - ord("1")]
        if found is not None:
            img, note = detect.crop_square(frame, found[1]), f"Saved {c}"
        else:
            img, note = frame, f"Saved {c} (full frame, no object found)"
        path = os.path.join(DATA_DIR, c, f"cam_{c}_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(path, img)   # clean image, no drawings
        message, message_until = note, time.time() + 1.0

cap.release()
cv2.destroyAllWindows()