# NBA Stadium Attendance Prediction Using Game Intensity Modeling

This project predicts NBA stadium attendance by modeling game intensity and combining it with contextual data like weather. A two-stage PyTorch pipeline first estimates point differential as a proxy for excitement, then predicts attendance to support scheduling, promotion, and dynamic pricing decisions for the NBA.

## 🧠 Key Features

- **Two-stage modeling pipeline**: Predict game score margin and use it to forecast attendance.
- **Data integration**: Combines box score, weather, and game metadata from 2022–2025 seasons.
- **Model serving**: FastAPI-based inference with ONNX optimization.
- **Online evaluation**: Data drift detection and synthetic data testing for real-time robustness.

## 🏀 Motivation

NBA game schedules and TV allocations are often fixed and rarely optimized for fan interest. This project introduces an ML-based tool that forecasts attendance using game intensity signals and contextual factors, allowing the NBA to identify underperforming matchups and optimize resource allocation, promotions, and ticket pricing.

## 🧪 System Overview

<p align="center">
  <img src="project_diagram.png" alt="System Diagram" width="100%">
</p>

1. **Model 1**: Predicts point differential using rolling averages of team statistics.
2. **Model 2**: Uses point differential and weather data to predict game attendance.
3. **Serving**: FastAPI endpoint for real-time inference; ONNX conversion for performance.
4. **Evaluation**: MLFlow for offline tracking, Alibi for online data drift detection.

## 📊 Evaluation Results

| Component         | Metric            | Value      |
|-------------------|-------------------|------------|
| Model 1           | RMSE / R²         | Tracked via MLFlow |
| Model 2           | RMSE / R²         | Tracked via MLFlow |
| Online Eval       | Response time     | ~1.2s avg |
| Drift Detection   | Alibi + Gaussian  | Detected expected changes |
| Final Output      | Attendance MAE    | Evaluated on 2024–25 games |

## 📦 Data Pipeline

- **Data Sources**: 
  - [nba_api](https://github.com/swar/nba_api) (box score, attendance)
  - [WeatherAPI](https://www.weatherapi.com) and [Open-Meteo](https://open-meteo.com) (temperature, wind, precipitation)
- **Time Range**: 2022–23 to 2024–25 regular seasons
- **Preprocessing**:
  - Rolling 5-game averages
  - Season-wise splits to prevent leakage
  - Weather joins by date and location

## ⚙️ Model Architecture

- **Model 1 (Score Prediction)**:
  - PyTorch MLP
  - Input: Team stats (rolling averages)
  - Output: Point differential
- **Model 2 (Attendance Prediction)**:
  - PyTorch MLP
  - Input: Output of Model 1 + weather data
  - Output: Predicted attendance
- **Optimizations**:
  - ONNX conversion
  - Graph and quantization optimizations for Model 2

## 🚀 Serving & Inference

- **Interface**: FastAPI web app
- **Inputs**: Game date, home team, away team
- **Flow**:
  1. Retrieve historical team stats and weather
  2. Run Model 1 → predict score diff
  3. Run Model 2 → predict attendance
- **Backend Support**:
  - .pth and ONNX model formats
  - Model staging, fallback logic
- **Limitations**:
  - Requires valid NBA team acronyms and dates
  - GPU-based performance tuning was partially constrained

## 🔍 Online Evaluation

- **Synthetic Data**:
  - Generated using Gaussian noise and future date shifts
  - [generate_online.py](./data_engineering/generate_online.py)
- **Drift Detection**:
  - Implemented with [Alibi Detect](https://github.com/SeldonIO/alibi-detect)
  - Reports confidence scores and alerts
  - [Notebook](./serving/online_eval/workspace/online_eval.ipynb)

## 🧱 Infrastructure

| Component            | Description                        |
|----------------------|------------------------------------|
| 2 × `m1.medium` VMs  | Training and serving environments  |
| A100 GPUs (x2)       | Model training with DDP            |
| Persistent Storage   | Object store for data and artifacts|
| Floating IPs         | Model endpoint + VM communication  |

## 🧰 Tech Stack

- Python, PyTorch, ONNX, FastAPI
- MLflow for experiment tracking
- Alibi for online evaluation
- Trovi + Chameleon Cloud (KVM@TACC) for compute

## 👥 Contributors

| Name            | Focus Area                        | Commits |
|------------------|-----------------------------------|---------|
| Will Calandra    | Model training                    | [Link](https://github.com/jasonmoon97/dynamic_nba_scheduling/commits/main/?author=wcalandra5) |
| Lake Wang        | Model serving, monitoring         | [Link](https://github.com/jasonmoon97/dynamic_nba_scheduling/commits/main/?author=Lake-Wang) |
| SungJoon Moon    | Data pipeline, infra provisioning | [Link](https://github.com/jasonmoon97/dynamic_nba_scheduling/commits/main/?author=jasonmoon97) |
| All Members      | Planning, integration, testing    | [Link](https://github.com/jasonmoon97/dynamic_nba_scheduling/commits/main/) |

## 📌 Future Improvements

- Full implementation of offline evaluation and load testing
- CI/CD integration for Ray-based job management
- Enhanced dashboarding for drift and performance monitoring
- Real-time feature enrichment (e.g., betting markets, injuries)

