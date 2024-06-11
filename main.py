import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

import model_data


app = Flask(__name__)
CORS(app)
@app.route('/predict_weight', methods=['POST'])
def predict_weight():
    data = request.json
    new_luggage = pd.DataFrame(data)
    new_luggage = new_luggage[model_data.X_train.columns]  # Make sure columns match the training data
    weight_predictions = model_data.model.predict(new_luggage)
    total_weight = sum(weight_predictions)
    response = {'total_weight': total_weight, 'weights': []}
    for i, weight in enumerate(weight_predictions):
        response['weights'].append({'barang': i + 1, 'weight': weight})
    
    max_weight_plane = 200 #kg
    if total_weight > max_weight_plane:
        response["status"] = "ditolak"
        response['pesan'] = "Beban melebihi ketentuan"
    else:
        response["status"] = "diterima"
        response['pesan'] = "Beban dibawah ketentuan dan pesawat boleh terbang"
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
