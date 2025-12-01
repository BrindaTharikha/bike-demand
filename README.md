# Bike Demand Forecasting API

LSTM-based microservice for predicting bike sharing demand 24 hours ahead.

## Model Performance

- MAE: 71.34 bikes
- RMSE: 111.50 bikes
- R²: 0.725
- Improvement over baseline: 26.4%

## Live API

**URL:** https://bike-demand-api.onrender.com (update after deployment)

**Documentation:** https://bike-demand-api.onrender.com/docs

**Health Check:** https://bike-demand-api.onrender.com/health

## Quick Start

### Test Health
```bash
curl https://bike-demand-api.onrender.com/health
```

### Get Prediction
```python
import requests
import numpy as np

url = "https://bike-demand-api.onrender.com/predict"

sample_data = {
    "last_48_hours": np.random.uniform(0, 1, (48, 15)).tolist()
}

response = requests.post(url, json=sample_data)
print(response.json())
```

## Input Format

48 hours of data, each hour with 15 features:

1. hr_sin (hour sine encoding)
2. hr_cos (hour cosine encoding)
3. weekday (0-6)
4. month_sin
5. month_cos
6. temp_celsius
7. humidity (0-1)
8. windspeed (0-1)
9. weathersit (1-4)
10. holiday (0-1)
11. workingday (0-1)
12. cnt_lag_1
13. cnt_lag_24
14. cnt_lag_168
15. cnt_rolling_24

## Output Format
```json
{
  "forecast_24h": [180.5, 220.3, ...],
  "avg_demand": 195.4,
  "peak_hour": 17,
  "peak_demand": 280.5,
  "min_demand": 15.2
}
```

## Technologies

- PyTorch LSTM (2 layers, 128 hidden units)
- FastAPI
- Deployed on Render
- 225,944 parameters

## Dataset

UCI Bike Sharing Dataset (17,379 hourly observations, 2011-2012)

## Author

Brinda - Northeastern University
Master's in Analytics
