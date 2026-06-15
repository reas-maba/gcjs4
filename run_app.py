import sys
import os

current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, os.path.join(current_dir, 'backend', 'src'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from model import DogBreedModel

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = DogBreedModel()
print("Model loaded successfully!")

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

@app.route('/')
def index():
    return jsonify({
        'message': 'DogBreeds Recognizer API',
        'endpoints': {
            'POST /api/predict': 'Upload image for breed prediction',
            'GET /api/breeds': 'Get list of all breeds',
            'GET /api/dog-info?breed=<name>': 'Get detailed info about a specific breed',
            'GET /api/all-dog-info': 'Get info about all breeds'
        }
    })

@app.route('/api/breeds', methods=['GET'])
def get_breeds():
    breeds = model.get_breed_list()
    return jsonify({'breeds': breeds})

@app.route('/api/dog-info', methods=['GET'])
def get_dog_info():
    breed_name = request.args.get('breed')
    if not breed_name:
        return jsonify({'error': 'breed parameter is required'}), 400
    
    info = model.get_dog_info(breed_name)
    return jsonify(info)

@app.route('/api/all-dog-info', methods=['GET'])
def get_all_dog_info():
    all_info = model.get_all_dog_info()
    return jsonify(all_info)

if __name__ == '__main__':
    print("Starting DogBreeds Recognizer API on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)