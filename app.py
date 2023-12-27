from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load('housing.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])

        # Prepare the input data for prediction
        input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]]
)
        # Make a prediction
        prediction = model.predict(input_data)
        print(prediction)
        # Render the prediction on a new page
        return render_template('prediciton.html', prediction=prediction[0])
    except Exception as e:
        return jsonify({'error': str('{e:e}')})

if __name__ == '__main__':
    app.run(debug=True)
