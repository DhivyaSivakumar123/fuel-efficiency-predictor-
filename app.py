from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('fuel_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        speed = float(request.form['average_speed'])
        petrol = float(request.form['petrol_used'])
        distance = float(request.form['distance_covered'])

        # Predict fuel efficiency using the model
        prediction = model.predict(np.array([[speed, petrol, distance]]))

        # CO₂ emission calculation: 2.31 kg CO₂ per liter of petrol
        co2_emission = petrol * 2.31

        return render_template(
            'result.html',
            prediction=round(prediction[0], 2),
            co2=round(co2_emission, 2)
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
