import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = torch.load(model_path, map_location=device)

    # If it's a state_dict, this will fail early
    if isinstance(model, dict):
        raise RuntimeError(
            "image_encoder.pt is a state_dict. "
            "Please load the full model object used during training."
        )

    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def predict_face(model, face_img):
    img = Image.fromarray(face_img)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

        # Handle different output shapes safely
        if output.numel() == 1:
            prob = torch.sigmoid(output).item()
        else:
            prob = torch.sigmoid(output.mean()).item()

    return round(prob, 2)
