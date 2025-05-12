from fastapi import FastAPI, Form
import os
import pandas as pd
import onnxruntime as ort
import numpy as np

app = FastAPI(
    title="NBA Game Prediction API (2-Stage ONNX)",
    description="Predicts NBA game score difference (model1) and attendance (model2)",
    version="1.0.0"
)

# === Load Models ===
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
model1_session = ort.InferenceSession("model1.onnx", providers=providers)
model2_session = ort.InferenceSession("model2.onnx", providers=providers)

# === Load Feature Data ===
base_data_dir = os.getenv("NBA_DATA_DIR", "nba_data")
X1_online = pd.read_csv(os.path.join(base_data_dir, "train/X_online_model1.csv"))
X2_online = pd.read_csv(os.path.join(base_data_dir, "train/X_online_model2.csv"))

# === Normalize Keys ===
def normalize(df):
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["home_team"] = df["home_team"].str.upper()
    df["away_team"] = df["away_team"].str.upper()
    return df

X1_online = normalize(X1_online)
X2_online = normalize(X2_online)


@app.post("/predict")
async def predict(date: str = Form(...), home_team: str = Form(...), away_team: str = Form(...)):
    try:
        game_date = pd.to_datetime(date).date()
        home_team = home_team.strip().upper()
        away_team = away_team.strip().upper()

        # === Lookup in X1 and X2 ===
        row1 = X1_online[
            (X1_online["game_date"] == game_date) &
            (X1_online["home_team"] == home_team) &
            (X1_online["away_team"] == away_team)
        ]
        row2 = X2_online[
            (X2_online["game_date"] == game_date) &
            (X2_online["home_team"] == home_team) &
            (X2_online["away_team"] == away_team)
        ]

        if row1.empty or row2.empty:
            return {"error": "No matching data found for input game."}

        # === Prepare model1 input ===
        input1 = row1.drop(columns=["game_date", "home_team", "away_team"]).values.astype(np.float32)
        input_name1 = model1_session.get_inputs()[0].name
        output1 = model1_session.run(None, {input_name1: input1})
        score_diff = output1[0]

        # === Prepare model2 input: concat(model1 output, X2 features) ===
        input2_raw = row2.drop(columns=["game_date", "home_team", "away_team"]).values.astype(np.float32)
        input2_combined = np.concatenate([input2_raw, score_diff], axis=1)
        input_name2 = model2_session.get_inputs()[0].name
        output2 = model2_session.run(None, {input_name2: input2_combined})
        attendance_prediction = float(output2[0][0])

        return {
            "predicted_score_difference": float(score_diff[0][0]) if score_diff.ndim == 2 else float(score_diff[0]),
            "predicted_attendance": attendance_prediction
        }

    except Exception as e:
        return {"error": str(e)}
