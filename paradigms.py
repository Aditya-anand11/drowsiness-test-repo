import torch

def hebbian_update(weights, x, y, lr=0.01):
    return weights + lr * torch.outer(y, x)

def ojas_update(weights, x, y, lr=0.01):
    decay = (y ** 2).unsqueeze(1) * weights
    return weights + lr * (torch.outer(y, x) - decay)

def competitive_learning(weights, x, lr=0.1):
    distances = torch.norm(weights - x, dim=1)
    winner_idx = torch.argmin(distances)
    weights[winner_idx] += lr * (x - weights[winner_idx])
    return weights, winner_idx.item()

def error_correction_update(weights, bias, x, y_true, lr=0.05):
    y_pred = 1.0 if (torch.dot(weights, x) + bias) >= 0 else 0.0
    error = y_true - y_pred
    weights = weights + lr * error * x
    bias = bias + lr * error
    return weights, bias, error

def energy_function(x, weights, bias):
    return -0.5 * torch.matmul(x, torch.matmul(weights, x)) - torch.dot(bias, x)

def energy_based_update(weights, bias, pos_data, neg_data, lr=0.01):
    pos_grad = torch.outer(pos_data, pos_data)
    neg_grad = torch.outer(neg_data, neg_data)
    weights += lr * (pos_grad - neg_grad)
    bias += lr * (pos_data - neg_data)
    return weights, bias
