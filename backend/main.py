import os
import io
import time
import gc
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Limit PyTorch to single thread to conserve memory on Render 512 MB RAM free tier
torch.set_num_threads(1)

app = FastAPI(
    title="Camelyon17 Research Demonstration Platform",
    description="Domain-Generalized Federated Learning for Robust Privacy-Preserving Medical Diagnosis Across Hospitals",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# Model definitions and benchmark metadata
MODEL_CONFIGS = {
    "GroupDRO": {
        "name": "GroupDRO",
        "title": "GroupDRO",
        "type": "Risk-aware federated optimization",
        "checkpoint": "GroupDRO_best.pt",
        "best_round_epoch": "Round 2",
        "test_accuracy": 94.2130,
        "balanced_accuracy": 93.8528,
        "precision": 97.8854,
        "recall": 89.3500,
        "f1": 93.4232,
        "auroc": 98.4044,
        "specificity": 98.3556,
        "fpr": 1.6444,
        "fnr": 10.6500,
        "confusion": {"TN": 43006, "FP": 719, "FN": 3967, "TP": 33282},
        "cost_seconds": 262.61,
        "privacy": "Standard FL",
        "validation_test_gap": 19.066,
        "is_best": True,
        "badge": "BEST RESEARCH MODEL",
        "description": "Group Distributionally Robust Optimization prioritizes worst-performing domain during training."
    },
    "ERM": {
        "name": "ERM",
        "title": "ERM",
        "type": "Centralized baseline",
        "checkpoint": "ERM_best.pt",
        "best_round_epoch": "Epoch 1",
        "test_accuracy": 93.8239,
        "balanced_accuracy": 93.5368,
        "precision": 96.3867,
        "recall": 89.9460,
        "f1": 93.0550,
        "auroc": 98.1697,
        "specificity": 97.1298,
        "fpr": 2.8702,
        "fnr": 10.0540,
        "confusion": {"TN": 42469, "FP": 1256, "FN": 3745, "TP": 33504},
        "cost_seconds": None,
        "privacy": "None (Centralized)",
        "validation_test_gap": 8.010,
        "is_best": False,
        "badge": None,
        "description": "Empirical Risk Minimization baseline trained on pooled data from Hospitals 0, 1, and 2."
    },
    "FedProx": {
        "name": "FedProx",
        "title": "FedProx",
        "type": "Federated proximal optimization",
        "checkpoint": "FedProx_best.pt",
        "best_round_epoch": "Round 1",
        "test_accuracy": 93.0756,
        "balanced_accuracy": 92.9275,
        "precision": 93.6948,
        "recall": 91.0763,
        "f1": 92.3670,
        "auroc": 97.6420,
        "specificity": 94.7788,
        "fpr": 5.2212,
        "fnr": 8.9237,
        "confusion": {"TN": 41442, "FP": 2283, "FN": 3324, "TP": 33925},
        "cost_seconds": 505.50,
        "privacy": "Standard FL",
        "validation_test_gap": 15.942,
        "is_best": False,
        "badge": None,
        "description": "Federated optimization with a proximal term to penalize client drift under domain shift."
    },
    "DP-WHFedDG": {
        "name": "DP-WHFedDG",
        "title": "DP-WHFedDG",
        "type": "Differentially private federated domain generalization",
        "checkpoint": "DP-WHFedDG_best.pt",
        "best_round_epoch": "Round 5",
        "test_accuracy": 80.0319,
        "balanced_accuracy": 80.7971,
        "precision": 72.7941,
        "recall": 90.3648,
        "f1": 80.6334,
        "auroc": 91.9413,
        "specificity": 71.2294,
        "fpr": 28.7706,
        "fnr": 9.6352,
        "confusion": {"TN": 31145, "FP": 12580, "FN": 3589, "TP": 33660},
        "cost_seconds": 1310.38,
        "privacy": "Differential Privacy (Clipping 1.0, Noise 0.8)",
        "validation_test_gap": 4.766,
        "is_best": False,
        "badge": "BEST VALIDATION-TEST STABILITY",
        "description": "Combines worst-hospital weighting, differential privacy (per-sample gradient clipping & noise), and federated learning."
    },
    "FedAvg": {
        "name": "FedAvg",
        "title": "FedAvg",
        "type": "Federated baseline",
        "checkpoint": "FedAvg_best.pt",
        "best_round_epoch": "Round 5",
        "test_accuracy": 73.2210,
        "balanced_accuracy": 75.0337,
        "precision": 63.6013,
        "recall": 97.6993,
        "f1": 77.0462,
        "auroc": 93.0683,
        "specificity": 52.3682,
        "fpr": 47.6318,
        "fnr": 2.3007,
        "confusion": {"TN": 22898, "FP": 20827, "FN": 857, "TP": 36392},
        "cost_seconds": 279.94,
        "privacy": "Standard FL",
        "validation_test_gap": 23.468,
        "is_best": False,
        "badge": "HIGHEST RECALL",
        "description": "Standard Federated Averaging algorithm across non-IID hospital client nodes."
    },
    "DP-FedAvg": {
        "name": "DP-FedAvg",
        "title": "DP-FedAvg",
        "type": "Differentially private federated averaging",
        "checkpoint": "DP-FedAvg_best.pt",
        "best_round_epoch": "N/A",
        "test_accuracy": None,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "auroc": None,
        "specificity": None,
        "fpr": None,
        "fnr": None,
        "confusion": None,
        "cost_seconds": None,
        "privacy": "Differential Privacy (Attempted)",
        "validation_test_gap": None,
        "is_best": False,
        "badge": "COMING SOON",
        "description": "Controlled baseline for DP without worst-hospital weighting. Implementation was numerically unstable in initial trials."
    }
}

# Loaded PyTorch models storage (kept empty; models are loaded on-demand to respect 512MB RAM)
loaded_models = {}

def get_resnet18_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    return model

def load_model_checkpoint(model_key: str):
    """
    Load a single model checkpoint on demand.
    Returns the loaded model or None if unavailable/failed.
    Releases temporary checkpoint dictionary and calls gc.collect() immediately.
    """
    if model_key not in MODEL_CONFIGS or model_key == "DP-FedAvg":
        return None

    cfg = MODEL_CONFIGS[model_key]
    ckpt_filename = cfg.get("checkpoint")
    if not ckpt_filename:
        return None

    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_filename)
    if not os.path.isfile(ckpt_path):
        return None

    try:
        model = get_resnet18_model()
        # weights_only=False is required because some checkpoints contain
        # numpy scalars (trusted source: own training outputs).
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        del checkpoint
        del state_dict
        gc.collect()
        return model
    except Exception as e:
        print(f"Error loading checkpoint for {model_key} from {ckpt_filename}: {e}")
        return None

# Standard evaluation transform
eval_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

def process_and_run_inference(image_bytes: bytes, model_key: str):
    if model_key == "DP-FedAvg":
        return {
            "name": "DP-FedAvg",
            "title": MODEL_CONFIGS["DP-FedAvg"]["title"],
            "type": MODEL_CONFIGS["DP-FedAvg"]["type"],
            "status": "coming_soon",
            "available": False,
            "reason": "Checkpoint not available. Implementation under future extension."
        }

    if model_key not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Model '{model_key}' is not recognized.")

    cfg = MODEL_CONFIGS[model_key]
    ckpt_filename = cfg.get("checkpoint")
    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_filename) if ckpt_filename else None
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise HTTPException(status_code=400, detail=f"Model '{model_key}' checkpoint is not available.")

    model = load_model_checkpoint(model_key)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint for {model_key}.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_width, orig_height = img.size
        resized = (orig_width != 96 or orig_height != 96)

        tensor = eval_transform(img).unsqueeze(0)  # [1, 3, 96, 96]

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            normal_prob = float(probs[0].item())
            anomalous_prob = float(probs[1].item())

        predicted_idx = 1 if anomalous_prob > normal_prob else 0
        predicted_class = "Anomalous" if predicted_idx == 1 else "Normal"
        confidence = float(max(normal_prob, anomalous_prob))

        return {
            "name": model_key,
            "title": cfg["title"],
            "type": cfg["type"],
            "status": "available",
            "available": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "normal_probability": normal_prob,
            "normal_percent": round(normal_prob * 100, 2),
            "anomalous_probability": anomalous_prob,
            "anomalous_percent": round(anomalous_prob * 100, 2),
            "original_size": [orig_width, orig_height],
            "target_size": [96, 96],
            "resized": resized,
            "badge": cfg.get("badge"),
            "is_best": cfg.get("is_best", False)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image for {model_key}: {str(e)}")
    finally:
        del model
        gc.collect()

# API Endpoints

@app.get("/api/health")
def get_health():
    available_models = [
        k for k, cfg in MODEL_CONFIGS.items()
        if k != "DP-FedAvg" and os.path.isfile(os.path.join(CHECKPOINTS_DIR, cfg.get("checkpoint", "")))
    ]
    pending_models = [k for k in MODEL_CONFIGS if k not in available_models]
    return {
        "status": "online",
        "service": "Camelyon17 Research Inference Engine",
        "pytorch_version": torch.__version__,
        "device": "CPU",
        "loaded_models_count": len(loaded_models),
        "loaded_models": list(loaded_models.keys()),
        "available_models": available_models,
        "pending_models": pending_models
    }

@app.get("/api/models")
def get_models():
    models_info = []
    for key, cfg in MODEL_CONFIGS.items():
        item = dict(cfg)
        ckpt_filename = cfg.get("checkpoint")
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_filename) if ckpt_filename else None
        item["available"] = bool(ckpt_path and os.path.isfile(ckpt_path) and key != "DP-FedAvg")
        models_info.append(item)
    return {"models": models_info}

@app.post("/api/predict")
async def predict_single(file: UploadFile = File(...), model_name: str = Form("GroupDRO")):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: PNG, JPG, JPEG, WEBP.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Limit is 10 MB.")

    result = process_and_run_inference(contents, model_name)
    return result

@app.post("/api/predict-all")
async def predict_all(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: PNG, JPG, JPEG, WEBP.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Limit is 10 MB.")

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_width, orig_height = img.size
    resized = (orig_width != 96 or orig_height != 96)
    tensor = eval_transform(img).unsqueeze(0)

    results = []
    for model_key in ["ERM", "FedAvg", "FedProx", "GroupDRO", "DP-WHFedDG", "DP-FedAvg"]:
        cfg = MODEL_CONFIGS[model_key]
        ckpt_filename = cfg.get("checkpoint")
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_filename) if ckpt_filename else None

        if model_key == "DP-FedAvg" or not ckpt_path or not os.path.isfile(ckpt_path):
            results.append({
                "name": model_key,
                "title": cfg["title"],
                "type": cfg["type"],
                "status": "coming_soon",
                "available": False,
                "reason": "Checkpoint not available",
                "badge": cfg.get("badge")
            })
            continue

        model = load_model_checkpoint(model_key)
        if model is None:
            results.append({
                "name": model_key,
                "title": cfg["title"],
                "type": cfg["type"],
                "status": "coming_soon",
                "available": False,
                "reason": "Checkpoint failed to load",
                "badge": cfg.get("badge")
            })
            continue

        try:
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                normal_prob = float(probs[0].item())
                anomalous_prob = float(probs[1].item())

            predicted_idx = 1 if anomalous_prob > normal_prob else 0
            predicted_class = "Anomalous" if predicted_idx == 1 else "Normal"
            confidence = float(max(normal_prob, anomalous_prob))

            results.append({
                "name": model_key,
                "title": cfg["title"],
                "type": cfg["type"],
                "status": "available",
                "available": True,
                "prediction": predicted_class,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 2),
                "normal_probability": normal_prob,
                "normal_percent": round(normal_prob * 100, 2),
                "anomalous_probability": anomalous_prob,
                "anomalous_percent": round(anomalous_prob * 100, 2),
                "original_size": [orig_width, orig_height],
                "target_size": [96, 96],
                "resized": resized,
                "badge": cfg.get("badge"),
                "is_best": cfg.get("is_best", False)
            })
        except Exception as e:
            print(f"Inference error for {model_key}: {e}")
            results.append({
                "name": model_key,
                "title": cfg["title"],
                "type": cfg["type"],
                "status": "coming_soon",
                "available": False,
                "reason": f"Inference error: {str(e)}",
                "badge": cfg.get("badge")
            })
        finally:
            del model
            gc.collect()

    del tensor
    gc.collect()

    return {
        "timestamp": time.time(),
        "input_info": {
            "filename": file.filename,
            "original_size": [orig_width, orig_height],
            "resized_to": [96, 96],
            "resized": resized,
            "size_bytes": len(contents)
        },
        "best_research_model": {
            "name": "GroupDRO",
            "reason": "Selected based on locked final unseen-hospital evaluation (Balanced Accuracy: 93.8528%, AUROC: 98.4044%). Not dynamically chosen per image."
        },
        "models": results
    }

@app.get("/api/research-results")
def get_research_results():
    return {
        "test_domain": "Hospital 4 (Unseen Hospital)",
        "test_samples": 80974,
        "test_patients": 9,
        "primary_ranking_metric": "Balanced Accuracy",
        "ranking": ["GroupDRO", "ERM", "FedProx", "DP-WHFedDG", "FedAvg", "DP-FedAvg"],
        "table": [
            {
                "method": "ERM",
                "accuracy": 93.8239,
                "balanced_accuracy": 93.5368,
                "precision": 96.3867,
                "recall": 89.9460,
                "f1": 93.0550,
                "auroc": 98.1697,
                "status": "Complete"
            },
            {
                "method": "FedAvg",
                "accuracy": 73.2210,
                "balanced_accuracy": 75.0337,
                "precision": 63.6013,
                "recall": 97.6993,
                "f1": 77.0462,
                "auroc": 93.0683,
                "status": "Complete"
            },
            {
                "method": "FedProx",
                "accuracy": 93.0756,
                "balanced_accuracy": 92.9275,
                "precision": 93.6948,
                "recall": 91.0763,
                "f1": 92.3670,
                "auroc": 97.6420,
                "status": "Complete"
            },
            {
                "method": "GroupDRO",
                "accuracy": 94.2130,
                "balanced_accuracy": 93.8528,
                "precision": 97.8854,
                "recall": 89.3500,
                "f1": 93.4232,
                "auroc": 98.4044,
                "status": "Complete (Best Overall)"
            },
            {
                "method": "DP-WHFedDG",
                "accuracy": 80.0319,
                "balanced_accuracy": 80.7971,
                "precision": 72.7941,
                "recall": 90.3648,
                "f1": 80.6334,
                "auroc": 91.9413,
                "status": "Complete (Best Stability)"
            },
            {
                "method": "DP-FedAvg",
                "accuracy": None,
                "balanced_accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "auroc": None,
                "status": "Coming Soon"
            }
        ],
        "metric_highlights": {
            "highest_accuracy": {"model": "GroupDRO", "value": "94.2130%"},
            "highest_balanced_accuracy": {"model": "GroupDRO", "value": "93.8528%"},
            "highest_precision": {"model": "GroupDRO", "value": "97.8854%"},
            "highest_recall": {"model": "FedAvg", "value": "97.6993%"},
            "highest_f1": {"model": "GroupDRO", "value": "93.4232%"},
            "highest_auroc": {"model": "GroupDRO", "value": "98.4044%"},
            "lowest_fpr": {"model": "GroupDRO", "value": "1.6444%"},
            "lowest_fnr": {"model": "FedAvg", "value": "2.3007%"}
        }
    }

@app.get("/api/generalization")
def get_generalization():
    return {
        "title": "Cross-Hospital Generalization Analysis",
        "training_hospitals": [0, 1, 2],
        "validation_hospital": 3,
        "unseen_test_hospital": 4,
        "validation_to_test_gap": [
            {"method": "DP-WHFedDG", "gap_percentage_points": 4.766, "badge": "SMALLEST VALIDATION-TO-TEST GAP"},
            {"method": "ERM", "gap_percentage_points": 8.010, "badge": None},
            {"method": "FedProx", "gap_percentage_points": 15.942, "badge": None},
            {"method": "GroupDRO", "gap_percentage_points": 19.066, "badge": None},
            {"method": "FedAvg", "gap_percentage_points": 23.468, "badge": None}
        ],
        "key_takeaway": "DP-WHFedDG showed the smallest validation-to-unseen-test performance gap (4.766 pp) in the completed experiments. Note: This demonstrates domain stability under noise/clipping, but does not imply highest absolute test accuracy."
    }

@app.get("/api/privacy")
def get_privacy():
    return {
        "method": "DP-WHFedDG",
        "clipping_norm": 1.0,
        "noise_multiplier": 0.8,
        "batch_size": 64,
        "local_epochs": 1,
        "federated_rounds": 5,
        "accounting_type": "Approximate analytical RDP estimate",
        "disclaimer": "The privacy accounting used in this study is an approximate analytical RDP estimate and is not presented as a formally certified end-to-end privacy guarantee."
    }

@app.get("/api/computational-cost")
def get_computational_cost():
    return {
        "costs": [
            {"method": "FedAvg", "round_time_seconds": 279.94, "relative_cost": "1.0x (Baseline)"},
            {"method": "GroupDRO", "round_time_seconds": 262.61, "relative_cost": "0.94x"},
            {"method": "FedProx", "round_time_seconds": 505.50, "relative_cost": "1.81x"},
            {"method": "DP-WHFedDG", "round_time_seconds": 1310.38, "relative_cost": "4.68x"}
        ],
        "explanation": "DP-WHFedDG required considerably more computation per round because per-sample gradient processing and Gaussian noise addition are required for the differential privacy mechanism."
    }

@app.get("/api/literature-context")
def get_literature_context():
    return {
        "disclaimer": "Published benchmark context — not directly comparable to this experimental protocol. Differences in architecture, pretraining, seeds (WILDS requires 10 seeds vs 1 seed here), and evaluation protocols materially affect results.",
        "published_wilds_benchmark": [
            {"method": "ContriMix", "model": "DenseNet121", "test_accuracy": "94.6%", "notes": "Standard WILDS benchmark submission"},
            {"method": "MBDG", "model": "DenseNet121", "test_accuracy": "93.3%", "notes": "Non-standard entry due to external pretraining"},
            {"method": "SGD Freeze-Embed", "model": "CLIP ViT-L", "test_accuracy": "96.5%", "notes": "Non-standard entry with massive vision-language pretraining"}
        ],
        "our_experimental_results": [
            {"method": "GroupDRO", "model": "ResNet18", "test_accuracy": "94.213%", "notes": "Single seed 42, focal loss"},
            {"method": "ERM", "model": "ResNet18", "test_accuracy": "93.824%", "notes": "Centralized baseline"},
            {"method": "FedProx", "model": "ResNet18", "test_accuracy": "93.076%", "notes": "Proximal FL"},
            {"method": "DP-WHFedDG", "model": "ResNet18", "test_accuracy": "80.032%", "notes": "DP + worst-hospital weighting"},
            {"method": "FedAvg", "model": "ResNet18", "test_accuracy": "73.221%", "notes": "Standard FL averaging"}
        ]
    }

@app.get("/api/experimental-notes")
def get_experimental_notes():
    return {
        "total_patch_dataset": 455954,
        "training_nodes": [0, 1, 2],
        "validation_node": 3,
        "unseen_test_node": 4,
        "final_test_samples": 80974,
        "final_test_patients": 9,
        "final_test_normal_count": 43725,
        "final_test_anomalous_count": 37249,
        "excluded_patients": [17, 20, 46, 86],
        "train_test_patient_overlap": 0,
        "val_test_patient_overlap": 0,
        "random_seed": 42,
        "image_dimensions": "96 x 96",
        "backbone": "ResNet18",
        "loss_function": "Focal Loss (gamma=2.0, alpha=0.25)",
        "optimizer": "SGD",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "federated_rounds": 5,
        "local_epochs": 1,
        "batch_size": 64
    }

# Mount static frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
