from fastapi import FastAPI, Form
import os
import pandas as pd
import torch
import numpy as np

app = FastAPI(
    title="NBA Game Prediction API (PyTorch)",
    description="Sequential prediction using two PyTorch models: score diff and attendance",
    version="1.0.0"
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch models
model1 = torch.load("score_diff_model.pth", map_location=device)
model1.to(device).eval()

model2 = torch.load("attendance_model.pth", map_location=device)
model2.to(device).eval()

# Load feature data
base_data_dir = os.getenv("NBA_DATA_DIR", "nba_data")
X1_online = pd.read_csv(os.path.join(base_data_dir, "train/X_online_model1.csv"))
X2_online = pd.read_csv(os.path.join(base_data_dir, "train/X_online_model2.csv"))

# Normalize keys
def normalize(df):
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["home_team"] = df["home_team"].str.upper()
    df["away_team"] = df["away_team"].str.upper()
    return df

X1_online = normalize(X1_online)
X2_online = normalize(X2_online)


@app.post("/predict")
async def predict(
    date: str = Form(...),
    home_team: str = Form(...),
    away_team: str = Form(...)
):
    try:
        game_date = pd.to_datetime(date).date()
        home_team = home_team.strip().upper()
        away_team = away_team.strip().upper()

        # Match rows in both CSVs
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

        # Prepare input for model1
        input1_np = row1.drop(columns=["game_date", "home_team", "away_team"]).values.astype(np.float32)
        input1_tensor = torch.from_numpy(input1_np).to(device)

        with torch.no_grad():
            score_diff_tensor = model1(input1_tensor)
            score_diff_val = score_diff_tensor.cpu().numpy()

        # Prepare input for model2
        input2_np = row2.drop(columns=["game_date", "home_team", "away_team"]).values.astype(np.float32)
        input2_combined = np.concatenate([input2_np, score_diff_val], axis=1)
        input2_tensor = torch.from_numpy(input2_combined).to(device)

        with torch.no_grad():
            attendance_tensor = model2(input2_tensor)
            attendance_val = float(attendance_tensor.cpu().numpy().squeeze())

        return {
            "predicted_score_difference": float(score_diff_val.squeeze()),
            "predicted_attendance": attendance_val
        }

    except Exception as e:
        return {"error": str(e)}
