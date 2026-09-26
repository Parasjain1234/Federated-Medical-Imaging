<<<<<<< HEAD
# Federated Medical Imaging
### Domain-Generalized Federated Learning for Robust Privacy-Preserving Medical Diagnosis Across Hospitals

> **Research Prototype — Not for clinical diagnostic use.**

An experimental framework for evaluating federated medical image classification under cross-hospital domain shift on the [Camelyon17](https://camelyon17.grand-challenge.org/) histopathology dataset. Combines worst-hospital risk-aware optimization and differential privacy with rigorous unseen-hospital evaluation.

---

## Overview

| Component | Description |
|-----------|-------------|
| **Dataset** | Camelyon17 (455,954 patches · 96 × 96 px · 5 hospital nodes) |
| **Backbone** | ResNet18 (ImageNet pretrained → fine-tuned, FC: 512 → 2) |
| **Loss** | Focal Loss (γ = 2.0, α = 0.25) |
| **Optimizer** | SGD (lr = 0.001, momentum = 0.9, wd = 1e-4) |
| **FL Setup** | 5 rounds, 1 local epoch, batch size 64 |
| **Train nodes** | Hospitals 0, 1, 2 |
| **Validation node** | Hospital 3 |
| **Test node** | Hospital 4 (unseen) |
| **Random seed** | 42 |

---

## Models Evaluated

| Method | Type | Privacy | Best Val. |
|--------|------|---------|-----------|
| **ERM** | Centralized baseline | None | Epoch 1 |
| **FedAvg** | Federated baseline | Standard FL | Round 5 |
| **FedProx** | Proximal FL (drift-aware) | Standard FL | Round 1 |
| **GroupDRO** | Risk-aware FL | Standard FL | Round 2 |
| **DP-WHFedDG** | Worst-hospital + DP | Gradient clipping + noise | Round 5 |
| **DP-FedAvg** | Differentially private FL | DP (attempted) | — (unstable) |

---

## Final Test Results — Unseen Hospital 4

> 80,974 patches · 9 patients · Patient-level leakage controlled (patients 17, 20, 46, 86 excluded)

| Method | Accuracy (%) | Balanced Acc. (%) | Precision (%) | Recall (%) | F1 (%) | AUROC (%) |
|--------|-------------|-------------------|--------------|------------|--------|-----------|
| **GroupDRO** ⭐ | **94.2130** | **93.8528** | **97.8854** | 89.3500 | **93.4232** | **98.4044** |
| ERM | 93.8239 | 93.5368 | 96.3867 | 89.9460 | 93.0550 | 98.1697 |
| FedProx | 93.0756 | 92.9275 | 93.6948 | 91.0763 | 92.3670 | 97.6420 |
| DP-WHFedDG | 80.0319 | 80.7971 | 72.7941 | 90.3648 | 80.6334 | 91.9413 |
| FedAvg | 73.2210 | 75.0337 | 63.6013 | **97.6993** | 77.0462 | 93.0683 |
| DP-FedAvg | N/A | N/A | N/A | N/A | N/A | N/A |

> **Primary ranking metric: Balanced Accuracy.**

### Key Findings

- **GroupDRO** achieved the strongest absolute performance across accuracy, balanced accuracy, F1, and AUROC.
- **DP-WHFedDG** showed the smallest validation-to-unseen-test performance gap (**4.766 pp**), indicating more stable domain transfer — despite lower absolute accuracy.
- **FedAvg** achieved the highest recall (**97.70%**) but also the highest false positive rate (**47.63%**), indicating a strong but poorly calibrated anomalous bias.
- **DP-FedAvg** suffered training instability (exploding gradients) and is reserved as a future extension.

### Validation → Test Stability Gap

| Method | Mean Abs. Gap (pp) |
|--------|-------------------|
| **DP-WHFedDG** ⭐ | **4.766** (smallest) |
| ERM | 8.010 |
| FedProx | 15.942 |
| GroupDRO | 19.066 |
| FedAvg | 23.468 |

---

## Privacy Configuration (DP-WHFedDG)

| Parameter | Value |
|-----------|-------|
| Gradient clipping norm | 1.0 |
| Noise multiplier (σ) | 0.8 |
| Local epochs | 1 |
| Federated rounds | 5 |
| Privacy accounting | Approximate analytical RDP |

> **Disclaimer:** The privacy accounting is an analytical RDP estimate and is **not** a formally certified end-to-end privacy guarantee.

---

## Computational Overhead

| Method | Avg. Round Time | Relative Cost |
|--------|----------------|--------------|
| GroupDRO | 262.61 s | 0.94× |
| FedAvg | 279.94 s | 1.00× (baseline) |
| FedProx | ~505.50 s | 1.81× |
| DP-WHFedDG | 1,310.38 s | **4.68×** |

DP-WHFedDG's overhead is due to per-sample gradient processing required by the differential privacy mechanism.

---

## Research Demonstration Platform

This repository includes a full **FastAPI + PyTorch + Vanilla JS** research web application.

### Running the Demo

```bash
# Clone repo
git clone https://github.com/pratham19092006/Federated-Medical-Imaging.git
cd Federated-Medical-Imaging

# Install dependencies (system Python recommended)
pip install -r requirements.txt

# Place model checkpoints in checkpoints/ directory:
# checkpoints/GroupDRO_best.pt
# checkpoints/ERM_best.pt
# checkpoints/FedAvg_best.pt
# checkpoints/FedProx_best.pt
# checkpoints/DP-WHFedDG_best.pt

# Launch server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Open browser at: http://localhost:8000
```

Or on Windows, double-click **`run.bat`**.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service status, loaded model count |
| `/api/models` | GET | All model configs and availability |
| `/api/predict` | POST | Single-model inference |
| `/api/predict-all` | POST | Multi-model inference (all available) |
| `/api/research-results` | GET | Full evaluation table + metric highlights |
| `/api/generalization` | GET | Validation-to-test gap analysis |
| `/api/privacy` | GET | DP configuration & accounting info |
| `/api/computational-cost` | GET | Training round times |
| `/api/literature-context` | GET | Published WILDS benchmark context |
| `/api/experimental-notes` | GET | Full dataset/hyperparameter specification |

### Checkpoint Structure

```
checkpoints/
├── GroupDRO_best.pt        ← Best research model (Round 2)
├── ERM_best.pt             ← Centralized baseline (Epoch 1)
├── FedAvg_best.pt          ← FL baseline (Round 5)
├── FedProx_best.pt         ← Proximal FL (Round 1)
├── DP-WHFedDG_best.pt      ← DP + worst-hospital (Round 5)
└── DP-FedAvg_best.pt       ← Reserved (not yet available)
```

> Checkpoint files are not included in this repository due to size (~43 MB each). Contact the author or retrain using the provided notebook.

---

## Literature Benchmark Context

> These values are **not directly comparable** to our results — different architectures, pretraining, number of seeds (WILDS requires 10; this study uses 1 seed), and protocols.

| Method | Architecture | Published Test Accuracy |
|--------|-------------|------------------------|
| ContriMix | DenseNet121 | 94.6% |
| MBDG | DenseNet121 | 93.3% |
| SGD Freeze-Embed (CLIP ViT-L) | CLIP ViT-L | 96.5% |

Our best result (GroupDRO, ResNet18, single seed): **94.21%**

---

## Limitations

1. **Single seed (42):** Statistical significance cannot be claimed.
2. **Approximate RDP accounting:** Not a formally certified privacy proof.
3. **DP-FedAvg:** Numerically unstable; excluded from final results.
4. **No strict ablation:** Privacy, domain weighting, and FL optimization are not individually isolated.
5. **Single dataset:** Generalization beyond Camelyon17 is not established.

---

## Repository Structure

```
├── backend/
│   └── main.py             # FastAPI server + PyTorch inference
├── frontend/
│   ├── index.html          # Research SPA (14 views)
│   ├── style.css           # Academic design system
│   └── app.js              # Client routing + inference + history
├── Camylon (2).ipynb       # Full training notebook
├── requirements.txt
├── run.bat                 # Windows startup script
└── run.ps1                 # PowerShell startup script
```

---

## Research Takeaway

> "GroupDRO achieved the strongest absolute predictive performance in the completed experiments, while DP-WHFedDG showed the smallest observed validation-to-unseen-test performance gap. These results suggest a trade-off between predictive performance, privacy-oriented optimization, domain stability, and computational cost."
>
> "DP-FedAvg remains a planned extension for isolating the incremental effect of differential privacy relative to standard FedAvg."

---

## Author

**Pratham Mishra**
Research project on federated domain generalization for medical imaging.

---

*Research Prototype · Not for clinical diagnosis*
=======
# Federated-Medical-Imaging
A privacy-preserving federated domain generalization framework for robust medical image classification across heterogeneous hospitals, combining worst-hospital risk awareness and differential privacy, with evaluation on an unseen hospital using Camelyon17.
>>>>>>> 8684d4167bddb8996d63a2e5539fec676eee84c3
