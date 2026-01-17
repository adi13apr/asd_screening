# vision/image_inference.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = "cpu"

# ---------------- BUILD SAME ENCODER ARCH ----------------
base_model = models.resnet18(weights=None)  # no pretrained, we load weights
encoder = nn.Sequential(
    *list(base_model.children())[:-1],
    nn.Flatten()
)

# ---------------- LOAD WEIGHTS ----------------
state_dict = torch.load("models/image_encoder.pt", map_location=DEVICE)
encoder.load_state_dict(state_dict)   # ✅ MATCHES EXACTLY
encoder.eval()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- FEATURE EXTRACTION ----------------
def extract_image_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = encoder(img)

    return features  # shape: (1, 512)
