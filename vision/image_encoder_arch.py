import torch.nn as nn
import torchvision.models as models

def get_image_encoder():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary output
    return model
