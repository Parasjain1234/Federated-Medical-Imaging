# 🧬 Federated Medical Imaging

### Domain-Generalized Federated Learning for Robust Privacy-Preserving Medical Diagnosis Across Hospitals

<p align="center">

  <a href="https://federated-medical-imaging-yfr9.onrender.com">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20Application-success?style=for-the-badge" alt="Live Demo">
  </a>

  <a href="https://github.com/Parasjain1234/Federated-Medical-Imaging">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>

</p>

<p align="center">

  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">

  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch">

  <img src="https://img.shields.io/badge/Federated%20Learning-Research-blue?style=for-the-badge" alt="Federated Learning">

  <img src="https://img.shields.io/badge/Medical%20Imaging-Research-purple?style=for-the-badge" alt="Medical Imaging">

</p>

> ⚠️ **Research Prototype — Not for clinical diagnostic use.**

An experimental framework for evaluating federated medical image classification under cross-hospital domain shift on the **Camelyon17** histopathology dataset. The project combines worst-hospital risk-aware optimization and differential privacy with evaluation on an unseen hospital domain.

---

# 🚀 Live Demo

## 🌐 Try the Research Platform

### 👉 [Open Federated Medical Imaging](https://federated-medical-imaging-yfr9.onrender.com)

The project includes a complete **FastAPI + PyTorch + Vanilla JavaScript** research demonstration platform for exploring trained federated learning models and their experimental results.

**Live Application:**  
https://federated-medical-imaging-yfr9.onrender.com

---

# 🔬 Overview

This project studies **federated medical image classification under cross-hospital domain shift** using the **Camelyon17** histopathology dataset.

The framework combines:

- 🏥 Federated Learning across hospital nodes
- 🌎 Domain Generalization for unseen hospitals
- ⚖️ Worst-hospital risk-aware optimization
- 🔐 Differential Privacy
- 🧠 ResNet18 deep learning backbone
- 📊 Patient-level leakage-controlled evaluation
- 🧪 Unseen-hospital testing
- 🌐 Interactive research demonstration platform

The primary experimental setup trains on hospitals **0, 1, and 2**, uses hospital **3** for validation, and evaluates generalization on the unseen **hospital 4**.

---

# 📊 Research Configuration

| Component | Description |
|-----------|-------------|
| **Dataset** | Camelyon17 |
| **Dataset Size** | 455,954 patches |
| **Image Size** | 96 × 96 px |
| **Hospital Nodes** | 5 |
| **Backbone** | ResNet18 |
| **Pretraining** | ImageNet |
| **Classifier** | 512 → 2 |
| **Loss** | Focal Loss |
| **Focal Loss γ** | 2.0 |
| **Focal Loss α** | 0.25 |
| **Optimizer** | SGD |
| **Learning Rate** | 0.001 |
| **Momentum** | 0.9 |
| **Weight Decay** | 1e-4 |
| **Federated Rounds** | 5 |
| **Local Epochs** | 1 |
| **Batch Size** | 64 |
| **Training Hospitals** | 0, 1, 2 |
| **Validation Hospital** | 3 |
| **Test Hospital** | 4 (Unseen) |
| **Random Seed** | 42 |

---

# 🤖 Models Evaluated

| Method | Type | Privacy | Best Validation |
|--------|------|---------|-----------------|
| **ERM** | Centralized baseline | None | Epoch 1 |
| **FedAvg** | Federated baseline | Standard FL | Round 5 |
| **FedProx** | Proximal FL | Standard FL | Round 1 |
| **GroupDRO** | Risk-aware FL | Standard FL | Round 2 |
| **DP-WHFedDG** | Worst-hospital + DP | Gradient clipping + noise | Round 5 |
| **DP-FedAvg** | Differentially private FL | DP attempted | Unstable |

---

# 🏥 Federated Learning Setup

```text
                         ┌──────────────────────┐
                         │   Federated Server   │
                         └──────────┬───────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
          ┌──────────┐        ┌──────────┐        ┌──────────┐
          │Hospital 0│        │Hospital 1│        │Hospital 2│
          └────┬─────┘        └────┬─────┘        └────┬─────┘
               │                   │                   │
               └───────────────────┼───────────────────┘
                                   │
                                   ▼
                         Model Aggregation
                                   │
                                   ▼
                         ┌────────────────┐
                         │ Global Model  │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │ Hospital 3     │
                         │ Validation     │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │ Hospital 4     │
                         │ Unseen Test    │
                         └────────────────┘
