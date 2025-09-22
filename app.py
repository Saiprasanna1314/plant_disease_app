# app.py
import os, numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = os.path.join(os.getcwd(), "plantvillage_cnn_64.h5")   # change if different
DATASET_DIR = r"C:\Users\sama\Documents\dataset\archive\plantvillage dataset\plantvillage_dataset_extracted\plantvillage dataset\color"
UPLOAD_FOLDER = os.path.join("static", "uploads")
IMG_SIZE = (64,64)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# load model once
model = load_model(MODEL_PATH)

# build labels from dataset folders (alphabetical, same order as flow_from_directory)
labels = [d for d in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, d))]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    f = request.files['file']
    if f.filename == '':
        return "No file selected", 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(save_path)

    img = load_img(save_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    label = labels[idx]
    prob = float(preds[idx])

    return render_template('index.html', prediction=label,  img_path=save_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or 5000 as fallback
    app.run(host="0.0.0.0", port=port, debug=True)
                 
