from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pickle

from CNN.CNN import *

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

current_directory = os.path.dirname(os.path.abspath(__file__))

file_name = "CNN/cnn_model_L0-14372585_R7973260.pkl"
file_path = os.path.join(current_directory, file_name)

with open(file_path, 'rb') as model_file:
    cnn = pickle.load(model_file)
cnn.alpha = 0.01

@app.route('/predict', methods=['POST'])
def predict():
    # with open('image_data.json', 'r') as json_file:
    #     image = np.array(json.load(json_file)).reshape((1, 28, 28))

    pixel_data = request.json.get('pixels')
    image = np.array(pixel_data, dtype="float64").reshape((1, 28, 28)) / 255

    prediction, probability, all_probs = cnn.predict(image)
    
    response = {
        'prediction': int(prediction),
        'probability': float(probability),
        'all_probs': all_probs.flatten().tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)