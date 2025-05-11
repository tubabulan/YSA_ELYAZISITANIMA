from flask import Flask, request, jsonify
import pytesseract
from PIL import Image

# Tesseract yolu belirt (gerekirse)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

@app.route('/recognize-text', methods=['POST'])
def recognize_text():
    image = request.files['image']
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return jsonify({"text": text})

if __name__ == '__main__':
    app.run(debug=True)
