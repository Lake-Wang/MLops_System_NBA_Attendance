import numpy as np
from flask import Flask, request, render_template_string
import tritonclient.http as httpclient
import os

app = Flask(__name__)

TRITON_SERVER_URL = os.environ['TRITON_SERVER_URL']
TABULAR_MODEL_NAME = os.environ['TABULAR_MODEL_NAME']

HTML_FORM = """
<!doctype html>
<title>Tabular Prediction</title>
<h2>Enter Features</h2>
<form method=post>
  Age: <input type=text name=age><br>
  Gender: <input type=text name=gender><br>
  Income: <input type=text name=income><br>
  <input type=submit value=Predict>
</form>
"""

def request_triton_tabular(age, gender, income):
    try:
        client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

        age = np.array([[float(age)]], dtype=np.float32)
        income = np.array([[float(income)]], dtype=np.float32)
        gender = np.array([[gender.encode('utf-8')]], dtype=object)  # if model expects BYTES

        inputs = []
        inputs.append(httpclient.InferInput("age", [1, 1], "FP32"))
        inputs.append(httpclient.InferInput("income", [1, 1], "FP32"))
        inputs.append(httpclient.InferInput("gender", [1, 1], "BYTES"))

        inputs[0].set_data_from_numpy(age)
        inputs[1].set_data_from_numpy(income)
        inputs[2].set_data_from_numpy(gender)

        outputs = [httpclient.InferRequestedOutput("prediction", binary_data=False)]

        results = client.infer(model_name=TABULAR_MODEL_NAME, inputs=inputs, outputs=outputs)
        prediction = results.as_numpy("prediction")[0][0]

        return prediction
    except Exception as e:
        print(f"Inference error: {e}")
        return "Error during inference"

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        income = request.form['income']
        prediction = request_triton_tabular(age, gender, income)
        return f"<h3>Prediction: {prediction}</h3>"
    return render_template_string(HTML_FORM)

@app.route('/test', methods=['GET'])
def test():
    return str(request_triton_tabular(age=30, gender='male', income=60000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
