from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle

app = FastAPI(title="Bike Demand Forecasting API")

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, forecast_length=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, forecast_length)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = LSTMForecaster()
model.load_state_dict(torch.load('best_improved_model.pth', map_location='cpu'))
model.eval()

with open('scaler_x.pkl', 'rb') as f:
    scaler_x = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

class PredictionInput(BaseModel):
    last_48_hours: list

@app.get("/")
def home():
    return {
        "service": "Bike Demand Forecasting API",
        "model": "LSTM with 15 features",
        "performance": {"MAE": 71.34, "R2": 0.725, "improvement": "26.4%"},
        "usage": "POST to /predict with 48 hours x 15 features"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        data = np.array(input_data.last_48_hours)
        
        if data.shape != (48, 15):
            raise HTTPException(400, f"Expected shape (48, 15), got {data.shape}")
        
        data_scaled = scaler_x.transform(data)
        x_tensor = torch.FloatTensor(data_scaled).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = model(x_tensor)
        
        prediction = scaler_y.inverse_transform(pred_scaled.numpy().reshape(-1, 1)).flatten()
        
        return {
            "forecast_24h": prediction.tolist(),
            "avg_demand": round(float(prediction.mean()), 2),
            "peak_hour": int(prediction.argmax()),
            "peak_demand": round(float(prediction.max()), 2),
            "min_demand": round(float(prediction.min()), 2)
        }
    except Exception as e:
        raise HTTPException(500, str(e))
