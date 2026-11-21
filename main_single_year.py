import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("key_west_2024.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (Rates of Change & Ratios)
# ---------------------------------------------------------
# Barometric pressure drop over 1-hour and 3-hour windows
df['pres_drop_1h'] = df['PRES'] - df['PRES'].shift(1)
df['pres_drop_3h'] = df['PRES'] - df['PRES'].shift(3)

# Wind gust-to-speed ratio and wind acceleration
df['gust_ratio'] = df['GST'] / (df['WSPD'] + 1e-5)
df['wind_accel_1h'] = df['WSPD'] - df['WSPD'].shift(1)

# Tide surge rate (change in water level over 2 hours)
df['tide_change_2h'] = df['TIDE'] - df['TIDE'].shift(2)

# Drop initial rows containing NaN from shifting
df_clean = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# 3. DEFINE FEATURES (X) AND TARGET (y)
# ---------------------------------------------------------
feature_cols = [
    'PRES', 
    'pres_drop_1h', 
    'pres_drop_3h', 
    'WSPD', 
    'GST', 
    'gust_ratio', 
    'wind_accel_1h', 
    'TIDE', 
    'tide_change_2h'
]

X = df_clean[feature_cols]
y = df_clean['storm_warning']

# Split chronologically (80% train, 20% test) without shuffling time order
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

# ---------------------------------------------------------
# 4. TRAIN XGBOOST MODEL
# ---------------------------------------------------------
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=25,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. EVALUATE & SAVE MODEL
# ---------------------------------------------------------
probs = model.predict_proba(X_test)[:, 1]
predictions = (probs >= 0.15).astype(int)

print("--- Classification Report ---")
print(classification_report(y_test, predictions, zero_division=0))

# Save trained model to disk for hardware/website integration
model.save_model("rhizocoast_model.json")
print("Model saved successfully as 'rhizocoast_model.json'")

# ---------------------------------------------------------
# 6. SIMULATE LIVE SENSOR INFERENCE (ALERT TEST)
# ---------------------------------------------------------
# Example live reading: rapidly dropping pressure (-4.2 hPa in 3h) & rising tide
live_reading = pd.DataFrame([{
    'PRES': 1004.0,
    'pres_drop_1h': -1.8,
    'pres_drop_3h': -4.2,
    'WSPD': 14.5,
    'GST': 19.8,
    'gust_ratio': 19.8 / 14.5,
    'wind_accel_1h': 2.5,
    'TIDE': 2.30,
    'tide_change_2h': 0.85
}])

storm_prob = model.predict_proba(live_reading)[0][1]
print(f"\nSimulated Live Storm Probability: {storm_prob * 100:.1f}%")

if storm_prob >= 0.75:
    print("ALERT STATUS: [LEVEL 2 WARNING] Evacuate / Notify Coastal Authorities!")
elif storm_prob >= 0.40:
    print("ALERT STATUS: [LEVEL 1 WATCH] Abnormal coastal conditions detected.")
else:
    print("ALERT STATUS: [NORMAL] Seawall sensors nominal.")