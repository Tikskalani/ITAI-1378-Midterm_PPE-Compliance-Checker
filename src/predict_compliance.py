"""
PPE Compliance Checker — standalone inference.

After training with notebooks/train_ppe_colab.ipynb you get a `best.pt`.
This script runs that model on an image (or a folder) and labels each worker
Compliant / Non-compliant, then saves annotated copies.

Usage:
    python src/predict_compliance.py --weights best.pt --source path/to/image_or_folder
    python src/predict_compliance.py --weights best.pt --source site.jpg --out out/ --conf 0.35

Compliance rule (per detected Person):
    NON-COMPLIANT if a `no_helmet` or `no_goggle` box overlaps the person,
    or if no `vest` box overlaps them. COMPLIANT otherwise.
Dataset classes (Ultralytics Construction-PPE):
    helmet, gloves, vest, boots, goggles, none, Person,
    no_helmet, no_goggle, no_gloves, no_boots
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO

VIOLATION_LABELS = {"no_helmet", "no_goggle"}


def center_in(box, person):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def annotate(model, img_path, conf=0.35):
    names = model.names
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    r = model(img_path, conf=conf, verbose=False)[0]
    dets = [(names[int(c)], list(map(float, b))) for c, b in zip(r.boxes.cls, r.boxes.xyxy)]
    persons = [b for n, b in dets if n.lower() == "person"]
    others = [(n, b) for n, b in dets if n.lower() != "person"]
    if not persons:  # fall back to an image-level judgement
        persons = [[0, 0, img.shape[1], img.shape[0]]]

    n_ok = n_bad = 0
    for p in persons:
        near = [n for n, b in others if center_in(b, p)]
        has_vest = any(n == "vest" for n in near)
        violations = [n for n in near if n in VIOLATION_LABELS]
        if not has_vest:
            violations.append("no_vest")
        ok = len(violations) == 0
        n_ok += ok
        n_bad += (not ok)
        color = (0, 170, 0) if ok else (0, 0, 220)  # BGR
        label = "COMPLIANT" if ok else "NON-COMPLIANT: " + ",".join(
            v.replace("no_", "no ") for v in violations
        )
        x1, y1, x2, y2 = map(int, p)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img, n_ok, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to trained best.pt")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", default="compliance_out", help="output folder")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    model = YOLO(args.weights)
    if os.path.isdir(args.source):
        paths = [p for p in sorted(glob.glob(os.path.join(args.source, "*")))
                 if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    else:
        paths = [args.source]

    os.makedirs(args.out, exist_ok=True)
    total_ok = total_bad = 0
    for p in paths:
        img, n_ok, n_bad = annotate(model, p, conf=args.conf)
        outp = os.path.join(args.out, os.path.basename(p))
        cv2.imwrite(outp, img)
        total_ok += n_ok
        total_bad += n_bad
        print(f"{os.path.basename(p)}: {n_ok} compliant, {n_bad} non-compliant -> {outp}")
    print(f"\nDone. {len(paths)} image(s). Totals: {total_ok} compliant, {total_bad} non-compliant.")


if __name__ == "__main__":
    main()
