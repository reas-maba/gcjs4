from flask import Flask, request, jsonify
from flask_cors import CORS
from model import DogBreedModel
import os

app = Flask(__name__)
CORS(app)

model = DogBreedModel()

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        result = model.predict(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/breeds', methods=['GET'])
def get_breeds():
    breeds = model.get_breed_list()
    return jsonify({'breeds': breeds})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)