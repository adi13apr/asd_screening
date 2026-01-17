import torch
from torchvision import models, transforms
from PIL import Image

model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_face_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        return model(img)
