import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import transforms
import joblib


# 📌 Model yapısı (eğitimdekiyle aynı olmalı)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 📌 Görseli işleyip tahmin yapan fonksiyon
def predict_image(image_path, model, encoder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    img = cv2.resize(img, (256, 64))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    img_tensor = transform(img).unsqueeze(0).float()

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = encoder.inverse_transform(predicted.numpy())[0]
        return label


# 📁 Dosya yolları (model klasörü içinde predict.py varsa)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "trained_model.pth")
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")
test_image_path = os.path.join(BASE_DIR, "..", "test", "test_inputs", "img4.png")  # test görseli

# 📥 Model ve encoder'ı yükle
encoder = joblib.load(encoder_path)
model = CNNModel(num_classes=len(encoder.classes_))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 🔍 Tahmin yap
prediction = predict_image(test_image_path, model, encoder)
print(f"🖋️ El yazısı tahmini: {prediction}")
