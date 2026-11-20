import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataloader import get_loaders
from models_cnn import ResNet, GoogLeNet

def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).int().cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().flatten())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

def benchmark_fps(model, input_shape=(1, 3, 64, 64), num_runs=50, device="cpu"):
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        elapsed = time.time() - start_time
    fps = num_runs / elapsed
    latency_ms = (elapsed / num_runs) * 1000
    return {"fps": fps, "latency_ms": latency_ms}

def compute_perclos(eye_states, threshold=0.8):
    closed_frames = np.sum(np.array(eye_states) == 0)
    perclos = closed_frames / len(eye_states)
    is_drowsy = perclos >= threshold
    return perclos, is_drowsy

if __name__ == "__main__":
    resnet = ResNet()
    googlenet = GoogLeNet()
    print("ResNet Benchmark:", benchmark_fps(resnet))
    print("GoogLeNet Benchmark:", benchmark_fps(googlenet))
