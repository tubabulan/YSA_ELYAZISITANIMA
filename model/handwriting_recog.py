import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

# Dataset sınıfı
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# CNN Modeli
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)  # 3 kanal RGB girişi
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 8, 512)  # Bu kısmı çıkış boyutuna göre ayarladık
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # Otomatik olarak batch dışındaki tüm boyutları düzleştirir
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Veriyi yükleme
def load_data(csv_path, image_folder, img_size=(256, 64)):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        img_name = row["Filenames"]
        label = row["Contents"]
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            print(f"Dosya bulunamadı: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Hatalı görsel: {img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Görseli RGB'ye dönüştür
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Dataloader hazırlığı
def prepare_data(images, labels, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB için normalizasyon
    ])

    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    test_dataset = CustomDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Model eğitimi
def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.float()
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

    # Test doğruluğu
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float()
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100*correct/total:.2f}%")

# 🔽 Dosya yollarını platformdan bağımsız yap
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(base_dir, "data", "Train", "train_labels.csv")
image_folder = os.path.join(base_dir, "data", "Train", "Images")

# 🔽 Veri yükle, encode et, model oluştur ve eğit
print(f"CSV dosyası mevcut mu? {os.path.exists(csv_path)}")
print(f"Image klasörü mevcut mu? {os.path.exists(image_folder)}")

images, labels = load_data(csv_path, image_folder)
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
train_loader, test_loader = prepare_data(images, labels)

# Modelin çıkış boyutunu kontrol et (dummy input ile)
dummy_input = torch.randn(1, 3, 256, 64)  # 1 resim, 3 kanal (RGB), 256x64 boyutunda
dummy_output = CNNModel(num_classes=len(np.unique(labels)))(dummy_input)
print(f"Modelin çıkış boyutu: {dummy_output.shape}")

model = CNNModel(num_classes=len(np.unique(labels)))
train_model(model, train_loader, test_loader)
##