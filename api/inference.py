import torch
import timm
from torchvision import transforms
from PIL import Image

CLASSES = [
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Pepper__bell___healthy",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Tomato_Early_blight",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Pepper__bell___Bacterial_spot",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Leaf_Mold",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Target_Spot",
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
    "Tomato_healthy",
    "Potato___healthy"
]

# model must match training EXACTLY
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=15)

checkpoint = torch.load(
    "saved_models/agrisustain_best.ckpt",
    map_location="cpu"
)

# 🔥 FIX: remove "model." prefix
state_dict = checkpoint["state_dict"]

new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("model.", "")  # remove Lightning wrapper
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image: Image.Image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "class": CLASSES[pred.item()],
        "confidence": float(conf.item())
    }