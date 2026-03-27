from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file and file.filename != "":
        # Create uploads folder if not exists
        os.makedirs("uploads", exist_ok=True)

        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img)
        pred_value = pred[0][0]

        # Result with confidence
        if pred_value > 0.5:
            result = f"Malignant ❌ ({pred_value:.2f})"
        else:
            result = f"Benign ✅ ({1 - pred_value:.2f})"

        return render_template('result.html', prediction=result)

    return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)