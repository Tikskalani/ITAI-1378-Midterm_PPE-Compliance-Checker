# PPE Compliance Checker

**Course:** ITAI 1378 — Computer Vision · Midterm Project
**Team:** Trilok Kalani · Matthew Jenkins
**Project Tier:** Tier 2 — fine-tuning a detector on a labeled dataset; the V2 capstone extends the same model to video.

> A computer-vision system that looks at a job-site image, detects each worker, and flags whether they are wearing the required safety gear — hard hat, hi-vis vest, and safety goggles — labeling each person "compliant" or "non-compliant."

## V1 Results (trained)
We trained the model on Google Colab (Tesla T4) for 40 epochs on the Construction-PPE dataset (1,416 images), which took about 17 minutes. On the held-out validation set it reached **mAP@50 = 0.612, mAP@50-95 = 0.302, precision 0.66, recall 0.59.** Training converged cleanly; the model is strong on common gear and weaker on the rare violation classes (`no_goggle`, `no_boots`). Full plots and the per-class breakdown are in `docs/results/`.

---

## 1. Problem Statement
On construction and industrial sites, workers who skip required PPE face some of the most common and most preventable injuries — head trauma, struck-by incidents, and eye injuries. Manual spot-checks by a safety officer don't scale: one person cannot watch every worker continuously, and violations are logged inconsistently. This is both a safety gap and an OSHA-compliance liability, and it is the problem our system addresses by assessing every worker in view the same way, every time.

## 2. Solution Overview
The system takes an image of a work area and runs a fine-tuned object detector that locates each person and each PPE item (hard hat, hi-vis vest, safety goggles). Per-worker logic then associates gear to the nearest person and outputs a bounding box plus a "compliant" / "non-compliant" status for each individual, based on which required items are and aren't detected. In V2 (the capstone), we extend this to a live or recorded video stream, tracking each worker across frames and logging compliance events with timestamps.

## 3. Technical Approach

| Element | Choice | Why |
|---|---|---|
| CV technique | Multi-class object detection | Locates and identifies people plus several gear types in a single pass. |
| Model | YOLO11 (Ultralytics), fine-tuned | Strong accuracy/speed trade-off, trains on a free Colab GPU, and is fast enough to carry into the V2 video extension. |
| Framework | PyTorch + Ultralytics | Standard, open-source, and includes built-in tracking (ByteTrack) for V2. |
| Compliance logic | Rule-based, per worker | Associate detected gear to the nearest person box; required set = hard hat + vest + goggles. |

## 4. Dataset Plan

| Source | Size | Labels | Link |
|---|---|---|---|
| Ultralytics Construction-PPE (used for V1) | 1,416 images (1,132 train / 143 val / 141 test) | 11 classes incl. helmet, vest, goggles, Person, no_helmet, no_goggle | https://docs.ultralytics.com/datasets/detect/construction-ppe/ |
| SH17 (V2 scale-up) | 8,099 images · ~75,994 instances | 17 PPE + body-part classes | https://github.com/ahmadmughees/SH17dataset |
| Roboflow Universe — PPE | Thousands of labeled images | person, hardhat, vest, goggles + NO- variants | https://universe.roboflow.com/browse/construction/ppe |

- **Classes used (V1):** `helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots`. The explicit `no_*` labels let the compliance rule flag violations directly from the detector.
- **Preparation:** standardize to YOLO format at 640×640, keep the provided train/val/test split, and apply light augmentation for site-lighting robustness.
- Datasets are public; each dataset's license is on its linked page.

## 5. Success Metrics

| Type | Metric | Target | V1 Actual |
|---|---|---|---|
| Primary | mAP@50 | ≥ 0.85 | **0.612** |
| Primary | mAP@50-95 | — | **0.302** |
| Supporting | Precision / Recall | — | 0.66 / 0.59 |
| Secondary | Inference latency | < 1 s / image | met on T4 |
| Secondary | Video throughput (V2) | ≥ 15 FPS | target |

The 0.85 mAP@50 was our target; V1 reached 0.61 on a small 1,416-image dataset, which is an honest baseline. The clearest path to improvement is more data (scaling to SH17), not a different metric.

## 6. System Diagram

```
[ Job-site image ]
        |
        v
[ YOLO11 detector ]  -- detects person + hard hat + vest + goggles
        |
        v
[ Per-worker compliance logic ]  -- associate gear -> nearest person
        |                           required set = {hard hat, vest, goggles}
        v
[ Output: box per worker + "Compliant" / "Non-compliant" ]

   V2: video -> track workers across frames -> timestamped compliance log
```

## 7. Week-by-Week Plan

| Week | Task | Milestone |
|---|---|---|
| 1 | Set up Colab; download + explore the PPE dataset | Dataset ready in YOLO format |
| 2 | Fine-tune YOLO11 on the PPE classes | Model trains, first mAP number |
| 3 | Add per-worker compliance logic; tune data | End-to-end compliant/non-compliant on images |
| 4 | Evaluate on held-out set; improve weak classes | Metrics measured (V1: mAP@50 0.61) |
| 5 | Build the demo; begin V2 video tracking (stretch) | Demo ready |
| 6 | Final testing, documentation, slides | Complete → present |

## 8. Challenges & Backup Plans — Risks & Mitigation

| Risk | Probability | Mitigation (Plan B) |
|---|---|---|
| Rare violation classes (`no_goggle`, `no_boots`) score low — confirmed in V1 (mAP@50 0.08–0.16) | High | More data for those classes: scale to SH17 (8,099 imgs) or oversample violation examples |
| Small dataset caps accuracy (V1 mAP@50 0.61) | High | Larger dataset + larger YOLO11 variant (n → s → m) + more epochs |
| Class imbalance (few `no_*` instances) | High | Weighted/focal loss, targeted augmentation, oversampling |
| Small / occluded goggles hard to detect | Medium | Higher input resolution + augmentation; scope to hat + vest if recall stays low |
| V2 video more complex than time allows | Medium | Keep V2 as the capstone/stretch; V1 (images) is the graded deliverable |

## 9. Resources Needed

| Resource | Notes | Cost |
|---|---|---|
| Compute | Google Colab (free T4 GPU) — V1 trained in ~17 min | $0 (Colab Pro optional) |
| Frameworks | PyTorch · Ultralytics (YOLO11) · OpenCV | $0 (open source) |
| Data / tools | Construction-PPE · SH17 · Roboflow (free) | $0 |

---

## Reproducing the results
The training runs in `notebooks/02_training.ipynb` on Google Colab with a T4 GPU. It fine-tunes YOLO11 on the Construction-PPE dataset (which downloads automatically), evaluates on the held-out set, and runs the compliance demo on sample images. The trained weights and evaluation plots are saved under `docs/results/`.

Run the trained model on any image:

```bash
pip install -r requirements.txt
python src/predict_compliance.py --weights docs/results/best.pt --source your_image.jpg --out out/
```

## Repository Structure

```
ITAI-1378-Midterm_PPE-Compliance-Checker/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_exploration.ipynb          data exploration + pretrained baseline
│   ├── 02_training.ipynb             training + evaluation + compliance demo
│   └── 03_training_results.ipynb     the run with outputs (mAP@50 0.61)
├── src/
│   └── predict_compliance.py         run the trained model on any image
├── data/
│   └── README.md                     dataset sources, classes, licenses
└── docs/
    ├── proposal.pdf                  presentation slides
    └── results/                      trained model, metric plots, per-class results
```

## References
- OSHA 29 CFR 1926.95–102 — PPE requirements for construction (head, eye, high-visibility).
- Ultralytics YOLO11 documentation and Construction-PPE dataset.
- Mughees et al., *SH17: A Dataset for Human Safety and PPE Detection in Manufacturing Industry*, arXiv:2407.04590 (2024).
- Roboflow Universe — PPE datasets.
