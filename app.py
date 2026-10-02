import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained model
MODEL_PATH = 'plant_disease_model.h5'
model = load_model(MODEL_PATH)

# Class labels (Must match the dataset folder names alphabetically)
CLASS_NAMES = ['Potato Early Blight', 'Potato Healthy', 'Potato Late Blight']

TREATMENTS = {
    'Potato Early Blight': 'Apply copper-based fungicides. Ensure crop rotation and avoid overhead sprinklers.',
    'Potato Late Blight': 'Apply systemic fungicides (e.g., Mancozeb or Ridomil). Remove and destroy infected plant debris.',
    'Potato Healthy': 'Your plant is healthy! Continue regular watering and proper soil management.'
}

def analyze_image(img_path):
    # Preprocess image for CNN model
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    confidence = float(np.max(predictions) * 100)
    class_idx = np.argmax(predictions)

    # Guardrail: If confidence is too low or prediction is uncertain
    CONFIDENCE_THRESHOLD = 85.0

    if confidence < CONFIDENCE_THRESHOLD:
        label = "Invalid / Non-Leaf Image Detected"
        remediation = "Please upload a clear, focused image of a potato plant leaf."
    else:
        label = CLASS_NAMES[class_idx]
        remediation = TREATMENTS.get(label, "Consult an agricultural expert.")

    return label, round(confidence, 2), remediation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html')

        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            label, confidence, remedy = analyze_image(file_path)
            
            return render_template(
                'index.html',
                label=label,
                confidence=confidence,
                remedy=remedy,
                image_path=file_path
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)