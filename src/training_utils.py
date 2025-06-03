# helper.py

import os
import torch
import torch.nn as nn
import logging
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ======= Model I/O =======

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model checkpoint saved to {path}")


def load_checkpoint(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    logger.info(f"Model checkpoint loaded from {path}")
    return model


# ======= Training & Evaluation =======

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Training loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    avg_loss = total_loss / len(dataloader)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    logger.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss, preds, targets


# ======= Metric Calculation =======

def compute_map(preds, targets):
    ap_list = []
    preds = preds.numpy()
    targets = targets.numpy()
    for i in range(targets.shape[1]):
        try:
            ap = average_precision_score(targets[:, i], preds[:, i])
        except ValueError:
            ap = 0.0
        ap_list.append(ap)
    mean_ap = sum(ap_list) / len(ap_list)
    logger.info(f"mAP@1: {mean_ap:.4f}")
    return mean_ap


# ======= Dataset Skeleton =======

class SoccerNetDataset(Dataset):
    def __init__(self, feature_dir, label_file, transform=None):
        self.feature_dir = feature_dir
        self.label_data = self._load_labels(label_file)
        self.transform = transform

    def _load_labels(self, label_file):
        # Placeholder: replace with CSV or JSON loading logic
        return []

    def __getitem__(self, idx):
        # Placeholder logic
        clip_features = torch.randn(1280, 15, 1, 1)  # Replace with actual loading
        labels = torch.randint(0, 2, (12,), dtype=torch.float32)
        return clip_features, labels

    def __len__(self):
        return len(self.label_data)


# ======= Inference Utility =======

def predict(model, input_clip, threshold=0.5):
    model.eval()
    with torch.no_grad():
        output = model(input_clip.unsqueeze(0))  # Add batch dimension
        pred = (output > threshold).int()
        logger.info(f"Predicted classes: {pred}")
        return pred
