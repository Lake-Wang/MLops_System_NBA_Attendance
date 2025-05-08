import requests
from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Set FastAPI backend URL (default to localhost for testing)
FASTAPI_SERVER_URL = os.environ.get('FASTAPI_SERVER_URL', 'http://localhost:8000')

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form.get('date')
    home_team = request.form.get('home_team', '').strip().upper()
    away_team = request.form.get('away_team', '').strip().upper()

    payload = {
        "date": date,
        "home_team": home_team,
        "away_team": away_team
    }

    try:
        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", data=payload)
        response.raise_for_status()
        result = response.json()

        prediction = {
            "score_diff": result.get("predicted_score_difference"),
            "attendance": result.get("predicted_attendance_category")
        }

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
