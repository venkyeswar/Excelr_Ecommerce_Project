from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os
import pandas as pd
import numpy as np

app = Flask(__name__)


model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':  

        features = [
            float(request.form['avg_session_length']),
            float(request.form['time_on_app']),
            float(request.form['time_on_website']),
            float(request.form['length_of_membership']),
        ]
        
        
        data = {
            "Avg Session Length":[features[0]],
            "Time on App":[features[1]],
            "Time on Website":[features[2]],
            "Length of Membership":[features[3]]
        }
        
        Inputs = pd.DataFrame(data)
        scaled_input = scaler.transform(Inputs)
        Inputs = pd.DataFrame(scaled_input,columns=Inputs.columns)


        prediction = model.predict(Inputs)
        prediction = np.round(prediction,3)
        prediction = abs(prediction)
        

        return render_template('result.html', features=features, prediction=prediction)

@app.route('/new_prediction')
def new_prediction():
    return redirect(url_for('index'))

if __name__ == "__main__":
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
     #app.run(debug = True)