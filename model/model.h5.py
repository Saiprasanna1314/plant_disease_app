from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model("model.h5")  # Make sure model.h5 exists in the same folder

# Home page
@app.route('/')
def home():
    return render_template('index.html')  # Make sure you have index.html in 'templates' folder

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'prediction': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'prediction': 'No file selected'})

    # Save uploaded file temporarily
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # Load image and preprocess (dummy preprocessing for now)
    img = image.load_img(filepath, target_size=(100,))  # Using input_shape=(100,) in dummy model
    img_array = image.img_to_array(img)
    img_array = img_array.flatten().reshape(1, 100)  # Flatten to match dummy input

    # Predict (dummy)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]

    # Map dummy class to label
    class_labels = ["No Disease", "Early Blight", "Leaf Curl", "Powdery Mildew", "Nitrogen Deficiency"]
    prediction = class_labels[pred_class % len(class_labels)]

    # Remove the uploaded file
    os.remove(filepath)

    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
