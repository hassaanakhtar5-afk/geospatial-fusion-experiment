# Experiment Code — Section 5

**Paper:** Intelligent Fusion of Multi-Source Geospatial Data for Complex Territorial Systems

## Structure

```
geospatial_fusion_experiment/
├── config.py                    # All hyperparameters and constants
├── main.py                      # Full Section 5 experiment pipeline
├── requirements.txt
├── data/
│   └── preprocessing.py         # Phases 1–2: ingestion, QA, harmonisation, PCA
├── models/
│   ├── classifiers.py           # RF, XGBoost, GeoViT classifier implementations
│   └── feedback.py              # Feedback operator Δ and harmonisation loop
├── evaluation/
│   └── metrics.py               # OA, Macro F1, Kappa, per-class, PR/ROC, McNemar
├── xai/
│   └── geoshap.py               # GeoSHAP, spatial sensitivity, attention backprojection
└── utils/
    ├── visualisation.py         # All paper figures (Fig. 3–17)
    └── reporting.py             # Tables 6–12 console output and text file
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Outputs (figures + results table) are saved to `outputs/`.

## What is covered

| Paper section | Code |
|---|---|
| 5.2 Experimental Setup | `data/preprocessing.py` — QA thresholds, PCA, stratified split |
| 5.3.1 GeoViT Architecture | `models/classifiers.py` — GeoViTModel with TransformerBlock, FocalLoss |
| 5.3.2 Classification Performance (Table 6) | `main.py::train_*` + `evaluation/metrics.py` |
| 5.3.3 Per-Class / High Risk (Table 7) | `evaluation/metrics.py::compute_per_class_metrics` |
| 5.3.4 Feedback Loop (Table 8) | `models/feedback.py::FeedbackLoop` |
| 5.3.5 Modality Importance (Fig. 7) | `xai/geoshap.py` + `utils/visualisation.py` |
| 5.3.6 PR and ROC curves (Fig. 8) | `evaluation/metrics.py` |
| 5.3.7 Ablation (Table 10) | `main.py::run_ablation_study` |
| 5.3.8 McNemar tests (Table 11) | `evaluation/metrics.py::mcnemar_test` |
| 5.4 Scalability | `data/preprocessing.py` — adaptive O(√n) grid, dual-threshold QA |
| 6.1 GeoSHAP (Figs. 11–14) | `xai/geoshap.py` |
| 6.2 Spatial Sensitivity (Fig. 15–16) | `xai/geoshap.py::compute_spatial_sensitivity_map` |
| 6.3 XAI Summary (Table 12, Fig. 17) | `utils/visualisation.py::plot_xai_radar_and_bars` |

## Notes

- Synthetic data faithfully replicates the Chelmsford SuDS dataset structure
  (35 modalities, 4 risk classes, 0.44% High Risk, cloud-masking nulls).
- No dummy or hardcoded metric values: all OA / F1 / Kappa figures are computed
  from actual model predictions on the held-out test set.
- Feedback loop loss curves are driven by real finite-difference gradient updates.
- GPU (CUDA) is used automatically when available; CPU fallback is supported.
