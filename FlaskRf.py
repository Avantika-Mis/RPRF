from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)


# Load the saved model from the pickle file
def load_model():
    with open('rainfall_model.pkl', 'rb') as model_file:
        return pickle.load(model_file)


model = load_model()


# Load the dataset and extract the actual district names
def load_districts():
    data = pd.read_csv(r'C:\Users\Saurav\Downloads\NNN.csv')
    data = data[['DISTRICT', 'Year', 'Actual(mm)']].dropna()

    # Encode DISTRICT column and create a mapping
    data['DISTRICT'] = data['DISTRICT'].astype('category')
    district_code_map = dict(enumerate(data['DISTRICT'].cat.categories))

    return district_code_map


district_code_map = load_districts()


@app.route('/')
def index():
    districts = list(district_code_map.values())  # List of actual district names
    return render_template('index.html', districts=districts)


@app.route('/predict', methods=['POST'])
def predict():
    district = request.form['district']
    date = request.form['date']

    # Extract year from date input
    year = datetime.strptime(date, '%Y-%m-%d').year

    # Get district code from district name (reverse mapping)
    district_code = {v: k for k, v in district_code_map.items()}.get(district, None)
    if district_code is None:
        return render_template('index.html', error="Invalid district selected", districts=district_code_map.values())

    # Prepare input features for the model
    input_features = np.array([[district_code, year]])

    # Predict rainfall using the model
    predicted_rainfall = model.predict(input_features)[0]

    # Provide activity recommendations based on the predicted rainfall
    if predicted_rainfall < 50:
        recommendation = "It's a good day for an outing."
    elif predicted_rainfall < 100:
        recommendation = "It might rain lightly, so plan outings accordingly."
    else:
        recommendation = "Heavy rainfall predicted, good day for indoor activities or farming."

    return render_template('index.html', prediction=predicted_rainfall, recommendation=recommendation,
                           districts=district_code_map.values())


if __name__ == '__main__':
    app.run(debug=True)