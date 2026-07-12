# Results — V1 trained model

**Run:** YOLO11s · 40 epochs · imgsz 640 · Google Colab Tesla T4 · ~17 min · Ultralytics 8.4.92
**Dataset:** Ultralytics Construction-PPE — 1,416 images (1,132 train / 143 val / 141 test), 11 classes.

## Headline metrics (best model on held-out val, 143 images / 1,172 instances)

| Metric | Value |
|---|---|
| mAP@50 | **0.612** |
| mAP@50-95 | **0.302** |
| mean precision | 0.663 |
| mean recall | 0.585 |

## Per-class mAP@50

| Class | mAP@50 | | Class | mAP@50 |
|---|---|---|---|---|
| Person | 0.533 | | none | 0.216 |
| vest | 0.532 | | no_helmet | 0.155 |
| boots | 0.450 | | no_boots | 0.085 |
| helmet | 0.427 | | no_goggle | 0.077 |
| gloves | 0.392 | | no_gloves | 0.077 |
| goggles | 0.380 | | | |

## Reading the results
- Training converged cleanly: losses fall smoothly and mAP plateaus around epoch 35–40, with no overfitting (`results_curves.png`).
- The model is strong on common gear — the confusion matrix (`confusion_matrix.png`) shows solid diagonals for helmet (169), vest (140), boots (116), Person (214), gloves (108).
- It is weaker on the rare violation classes (`no_goggle`, `no_gloves`, `no_boots`), which have few training examples. This matters because the compliance rule relies on `no_helmet` / `no_goggle`.
- The improvement path is more data for those classes — scaling to SH17 (8,099 images) or oversampling violation examples. This is the V2 work.

## Files
- `results_curves.png` — training/val loss + precision/recall/mAP curves
- `confusion_matrix.png` — per-class confusion matrix
- `best.pt` — trained model weights
- `compliance_out/` — sample images labeled Compliant / Non-compliant
