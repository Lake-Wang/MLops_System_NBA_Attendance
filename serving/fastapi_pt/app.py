from fastapi import FastAPI, Form
from pydantic import BaseModel
import torch
import torch.nn.functional as F

app = FastAPI(
    title="NBA Game Prediction API",
    description="Sequential prediction using two models: score diff and attendance",
    version="1.0.0"
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model1 = torch.load("score_diff_model.pth", map_location=device)
model1.to(device).eval()

model2 = torch.load("attendance_model.pth", map_location=device)
model2.to(device).eval()

# Example team encoding (simple version)
TEAM_ENCODINGS = {
    "LAL": 0, "BOS": 1, "GSW": 2, "MIA": 3, "NYK": 4,  # etc.
    # Include all teams used in training
}

@app.post("/predict")
async def predict(
    date: str = Form(...),
    home_team: str = Form(...),
    away_team: str = Form(...)
):
    try:
        # Convert teams to numeric encoding
        if home_team not in TEAM_ENCODINGS or away_team not in TEAM_ENCODINGS:
            return {"error": "Invalid team abbreviation."}

        home_enc = TEAM_ENCODINGS[home_team]
        away_enc = TEAM_ENCODINGS[away_team]

        # Feature tensor for model1: e.g., [home_team_id, away_team_id]
        input1 = torch.tensor([[home_enc, away_enc]], dtype=torch.float32).to(device)

        # Predict score difference
        with torch.no_grad():
            score_diff = model1(input1)
            score_diff_val = int(score_diff.item())  # Ensure it's an int

        # Prepare input for model2: could be just score_diff or combined features
        input2 = torch.tensor([[score_diff_val]], dtype=torch.float32).to(device)

        # Predict attendance category
        with torch.no_grad():
            attendance_pred = model2(input2)
            attendance_val = int(torch.round(attendance_pred).item())

        return {
            "predicted_score_difference": score_diff_val,
            "predicted_attendance_category": attendance_val
        }

    except Exception as e:
        return {"error": str(e)}
