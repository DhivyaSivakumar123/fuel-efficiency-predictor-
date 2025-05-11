from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model (ensure the model file is in the same directory as app.py)
model = joblib.load('fuel_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Renders the form page

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the form
            speed = float(request.form['average_speed'])
            petrol = float(request.form['petrol_used'])
            distance = float(request.form['distance_covered'])

            # Make the prediction
            prediction = model.predict(np.array([[speed, petrol, distance]]))
            
            # Render the result page with the prediction
            return render_template('result.html', prediction=prediction[0])
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
