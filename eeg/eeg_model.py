import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- EEG MODEL ----------------
class EEGNet(nn.Module):
    def __init__(self, channels=19):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ---------------- LOAD MODEL ----------------
eeg_encoder = EEGNet().to(DEVICE)

state_dict = torch.load(
    "models/eeg_encoder.pt",
    map_location=DEVICE
)

eeg_encoder.load_state_dict(state_dict)
eeg_encoder.eval()


# ---------------- PREPROCESSING ----------------
def normalize_eeg(eeg):
    mean = eeg.mean(dim=1, keepdim=True)
    std = eeg.std(dim=1, keepdim=True) + 1e-6
    return (eeg - mean) / std


def window_eeg(eeg, window_size=256, stride=256):
    windows = []
    T = eeg.shape[1]

    for start in range(0, T - window_size, stride):
        windows.append(eeg[:, start:start + window_size])

    return torch.stack(windows)


# ---------------- FEATURE EXTRACTION ----------------
def extract_eeg_features(csv_path):
    """
    csv_path : path to EEG CSV file
    return   : numpy array (32,)
    """

    df = pd.read_csv(csv_path, header=None)
    data = df.values.astype(np.float32)

    eeg = torch.tensor(data).T               # (channels, time)
    eeg = normalize_eeg(eeg)
    windows = window_eeg(eeg)                # (N, channels, time)

    windows = windows.to(DEVICE)

    with torch.no_grad():
        embeddings = eeg_encoder(windows)    # (N, 32)

    # Average across windows
    features = embeddings.mean(dim=0)

    return features.unsqueeze(0)
