# main.py
import pandas as pd
from models.forecast_model import forecast_next
from models.attribution_model import load_or_train, predict_breakdown

# Load dataset
hist = pd.read_csv("data/flights_2025.csv")

# Forecast next month
target_month = '2025-05'
avg_delay = forecast_next(hist, target_month)
print(f"⏳ Avg Delay: {avg_delay} minutes per flight")

# Train attribution model
carrier_code = "9E"
airport_code = "ABE"
aobj = load_or_train(hist, carrier=carrier_code, airport=airport_code)

# Prepare row for breakdown (latest valid row)
row = hist.dropna(subset=aobj['features']).iloc[-1]
breakdown = predict_breakdown(aobj, row)
print("📊 Breakdown:", breakdown)

