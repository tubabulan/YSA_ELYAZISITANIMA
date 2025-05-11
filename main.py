# main.py

from model import DENEME  # predict_image ve CNNModel burada
from model.handwriting_recog import ocr, select_image
import os
import joblib
import torch

def main():
    print("📌 Görsel seçiliyor...")
    file_path = select_image()
    if not file_path:
        print("❌ Görsel seçilmedi.")
        return

    print("📸 OCR işlemi başlıyor...")
    ocr(file_path)

    print("🧠 Eğitilmiş model yükleniyor...")
    # Model ve encoder yolları
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "trained_model.pth")
    encoder_path = os.path.join(base_dir, "model", "label_encoder.pkl")

    # Model ve encoder yükleniyor
    encoder = joblib.load(encoder_path)
    model = neww.CNNModel(num_classes=len(encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    print("🔍 El yazısı sınıflandırması yapılıyor...")
    prediction = neww.predict_image(file_path, model, encoder)
    print(f"✅ Tahmin sonucu: {prediction}")

if __name__ == "__main__":
    main()
