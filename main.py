import torch
from vision.blink_detection import get_blink_features
from vision.gesture_detection import get_gesture_features
from eeg.eeg_model import extract_eeg_features
from fusion.multimodal_model import ASDMultimodalModel
from vision.image_inference import extract_image_features

image = extract_image_features("data/user_image.jpg").float()
blink =torch.tensor(get_blink_features(show=True,duration=15)).float().unsqueeze(0)
gesture =torch.tensor(get_gesture_features(max_seconds=15)).float().unsqueeze(0)


eeg = extract_eeg_features("data/sample_eeg.csv")
eeg = eeg.float()


# FIX: ensure shape (1, 32)
if eeg.dim() == 1:
    eeg = eeg.unsqueeze(0)
elif eeg.dim() == 3:
    eeg = eeg.mean(dim=-1)
print("blink:", blink.shape)
print("gesture:", gesture.shape)
print("eeg:", eeg.shape)

model = ASDMultimodalModel()
model.eval()
with torch.no_grad():
    risk = model(blink, gesture, eeg,image)


risk_percent = risk.item() * 100
print(f"ASD Risk: {risk_percent:.1f}%")