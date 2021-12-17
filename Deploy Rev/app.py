import numpy as np
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)
model = load('model.joblib')

def text_format(string):
    length = len(string)
    to_print = ''
    for i in range(2, length-2):
        to_print += string[i]
    
    return to_print

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = text_format(str(prediction))

    return render_template('index.html', prediction_text = 'Prediksi Kepadatan Penduduk: {}'.format(output)) #print formastting

if __name__ == '__main__':
    app.run(debug=True)