from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle
from lightgbm import LGBMRegressor

app = Flask(__name__)

with open('model\\burn_rate_model', 'rb') as f:
    model = pickle.load(f)

with open('model\\burn_rate_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        gender = float(request.form['gender'])
        type = float(request.form['type'])
        wfh = float(request.form['wfh'])
        designation = float(request.form['Designation'])
        ra = float(request.form['RA'])
        mfs = float(request.form['MFS'])
    
    rate = 0

    data = ((gender,type,wfh,designation,ra,mfs),)
    arr = np.array(data, dtype=np.float32)
    x = scaler.transform(arr)
    predict = model.predict(x)
    rate = predict[0]
    rate=round(rate,2)
    return render_template('index.html', rate=rate)

if __name__ == '__main__':
    app.run(debug=True)