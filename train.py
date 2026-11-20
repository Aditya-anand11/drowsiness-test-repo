import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_loaders, get_sequence_loader
from models_cnn import CNN2D
from models_temporal import TemporalLSTM
from models_vae import VAE, vae_loss_fn

def set_seed(seed=42):
    """Seed all RNGs so training is reproducible regardless of external state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN where available for stable runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def get_optimizer(model, opt_name="adam", lr=0.001, l2=0.0):
    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    elif opt_name == "momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2)
    elif opt_name == "nesterov":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=l2)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

def train_ablation_cnn(dataset_type="both", opt_name="adam", use_bn=True, dropout=0.5, l2=0.001, init_type="he", augment=True, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(dataset_type=dataset_type, augment=augment)
    model = CNN2D(use_bn=use_bn, dropout=dropout, init_type=init_type).to(device)
    optimizer = get_optimizer(model, opt_name=opt_name, lr=0.001, l2=l2)
    criterion = nn.BCELoss()

    if len(train_loader) == 0:
        raise ValueError("Empty training loader; check dataset directories.")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} Acc: {acc:.4f}")
    return model

def train_temporal(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_sequence_loader()
    model = TemporalLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    if len(loader) == 0:
        raise ValueError("Empty sequence loader.")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Temporal Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")
    return model

def train_vae(dataset_type="both", epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_loaders(dataset_type=dataset_type, augment=False)
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if len(train_loader) == 0:
        raise ValueError("Empty training loader; check dataset directories.")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(images)
            loss = vae_loss_fn(recon_x, images, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"VAE Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")
    return model
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss = vae_loss_fn(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"VAE Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")
    return model

if __name__ == "__main__":
    model = train_ablation_cnn(dataset_type="both", opt_name="momentum", use_bn=True, augment=True, epochs=2)
