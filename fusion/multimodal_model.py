# fusion/multimodal_model.py

import torch
import torch.nn as nn

class ASDMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(34, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, blink, gesture, eeg):
        if eeg.dim() == 3:
            eeg = eeg.mean(dim=-1)
        x = torch.cat([blink, gesture, eeg], dim=1)
        return self.fc(x)
