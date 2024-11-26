from flask import Flask, request, jsonify
from ultralytics import YOLO
import io
from PIL import Image

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO('best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        # Load image
        image = Image.open(io.BytesIO(file.read()))

        # Perform detection using YOLOv8
        results = model.predict(image)

        # Process detection results
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': model.names[int(box.cls)],  # Class name (e.g., Burung, Tikus)
                    'confidence': float(box.conf)        # Confidence score
                })

        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)