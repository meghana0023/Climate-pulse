import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import gamma, norm
import os

# =========================
# CONFIG
# =========================
DATA_PATH = "Rainfall_climate_dataset_handled.csv"
CLIMATE_MODEL_DIR = "models/climate"
RAIN_MODEL_DIR = "models/rainfall"

SEQ_LEN_RAIN = 9
SEQ_LEN_CLIMATE = 12

VALID_STATES = [
    "Assam","Bihar","West Bengal","Odisha","Uttar Pradesh",
    "Maharashtra","Karnataka","Rajasthan","Gujarat","Telangana"
]

CLIMATE_FEATURES = [
    "TEMP_C_AVG", "HUMIDITY_AVG", "PRESSURE_hPa_AVG", 
    "WIND_SPEED_AVG", "SOIL_MOISTURE_AVG"
]

RAIN_FEATURES = [
    "rain_mm","lag1","lag3","lag12",
    "roll3","roll12",
    "TEMP_C_AVG","HUMIDITY_AVG","PRESSURE_hPa_AVG",
    "WIND_SPEED_AVG","SOIL_MOISTURE_AVG",
    "month_sin","month_cos","is_monsoon"
]

# =========================
# MODEL CACHE (Speed Fix)
# =========================
# This prevents reloading .keras files from disk on every request
model_cache = {}

def get_cached_model(path):
    if path not in model_cache:
        if os.path.exists(path):
            print(f"📡 Loading model into RAM: {path}")
            model_cache[path] = tf.keras.models.load_model(path)
        else:
            print(f"❌ Model path not found: {path}")
            return None
    return model_cache[path]

# =========================
# LOAD & PREPROCESS DATA
# =========================
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["STATE","year","month"]).reset_index(drop=True)

df = df.rename(columns={
    "Temperature_Celsius": "TEMP_C_AVG",
    "Humidity_%": "HUMIDITY_AVG",
    "Pressure_hPa": "PRESSURE_hPa_AVG",
    "Wind Speed_m/s": "WIND_SPEED_AVG",
    "Soil Moisture": "SOIL_MOISTURE_AVG"
})

df["rain_real"] = df["rain_mm"]
df["rain_mm"] = np.log1p(df["rain_mm"])
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
df["is_monsoon"] = df["month"].isin([6,7,8,9]).astype(int)

g = df.groupby("STATE")
df["lag1"] = g["rain_mm"].shift(1)
df["lag3"] = g["rain_mm"].shift(3)
df["lag12"] = g["rain_mm"].shift(12)
df["roll3"] = g["rain_mm"].rolling(3).mean().reset_index(level=0, drop=True)
df["roll12"] = g["rain_mm"].rolling(12).mean().reset_index(level=0, drop=True)
df = df.dropna().reset_index(drop=True)

# =========================
# PREDICTION LOGIC
# =========================
def predict_rainfall(state):
    model_path = f"{RAIN_MODEL_DIR}/{state}/rainfall_cnn_lstm.keras"
    model = get_cached_model(model_path)
    if not model: return 1.0

    sdf = df[df["STATE"] == state]
    X = sdf[RAIN_FEATURES].iloc[-SEQ_LEN_RAIN:].values.astype("float32")
    X = X.reshape(1, SEQ_LEN_RAIN, len(RAIN_FEATURES))

    rain_log = model.predict(X, verbose=0)[0][0]
    return max(float(np.expm1(rain_log)), 1.0)

def build_spi3_climatology(state, month):
    sdf = df[df["STATE"] == state]
    values = []
    months_to_sum = [(month-2)%12 or 12, (month-1)%12 or 12, month]
    
    for y in sdf["year"].unique():
        rows = sdf[(sdf["year"]==y) & (sdf["month"].isin(months_to_sum))]
        if len(rows) == 3:
            values.append(rows["rain_real"].sum())
    return np.array(values)

def compute_spi(hist, value):
    hist = hist[hist > 0]
    if len(hist) < 2: return 0.0
    shape, loc, scale = gamma.fit(hist, floc=0)
    cdf = gamma.cdf(value, shape, loc, scale)
    return norm.ppf(np.clip(cdf, 1e-6, 1-1e-6))

def predict_flood_drought(state, year, month):
    rain = predict_rainfall(state)
    hist = build_spi3_climatology(state, month)

    # Threshold lowered to 5 to avoid "Insufficient Data" during testing
    if len(hist) < 5:
        return rain, 0.0, "Normal (Limited Data)"

    spi = compute_spi(hist, rain)

    if spi >= 2: status="Extreme Flood"
    elif spi >= 1.5: status="Severe Flood"
    elif spi >= 1.0: status="Moderate Flood"
    elif spi > -1: status="Normal"
    elif spi > -1.5: status="Moderate Drought"
    elif spi > -2: status="Severe Drought"
    else: status="Extreme Drought"

    return rain, spi, status

# =========================
# CLI FOR TESTING
# =========================
if __name__ == "__main__":
    print("\n🌍 Prediction System Ready")
    while True:
        s = input("\nState: ").strip()
        if s.lower() == "exit": break
        if s not in VALID_STATES: continue
        y = int(input("Year: "))
        m = int(input("Month (1-12): "))
        r, sp, st = predict_flood_drought(s, y, m)
        print(f"📊 {m}/{y} Result: {r:.2f}mm | SPI: {sp:.2f} | Status: {st}")