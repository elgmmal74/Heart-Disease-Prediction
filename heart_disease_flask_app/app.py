from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Create the Flask app
app = Flask(__name__)

# Load the compressed model
model = joblib.load('model/heart_disease_classification_model_compressed.pkl')

# Main page with input form
@app.route('/')
def home():
    return render_template('index.html')

# Predict result based on user input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Arrange the data in an array for prediction
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Predict using the loaded model
        prediction = model.predict(features)

        # Determine the result
        result = 'High risk of heart disease' if prediction[0] == 1 else 'Low risk of heart disease'

        # Display the result
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
