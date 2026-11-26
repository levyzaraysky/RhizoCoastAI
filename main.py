import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. LOAD MULTI-YEAR DATA
# ---------------------------------------------------------
df = pd.read_csv("key_west_2022_2024.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
df['pres_drop_1h'] = df['PRES'] - df['PRES'].shift(1)
df['pres_drop_3h'] = df['PRES'] - df['PRES'].shift(3)
df['gust_ratio'] = df['GST'] / (df['WSPD'] + 1e-5)
df['wind_accel_1h'] = df['WSPD'] - df['WSPD'].shift(1)
df['tide_change_2h'] = df['TIDE'] - df['TIDE'].shift(2)

df_clean = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# 3. YEAR-BASED TRAIN / TEST SPLIT (No Seasonal Bias!)
# ---------------------------------------------------------
data_used = [
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

# Train on 2022-2023 (all 4 seasons x 2), Test on unseen 2024
train_mask = df_clean['timestamp'].dt.year < 2024
test_mask = df_clean['timestamp'].dt.year == 2024

X_train, y_train = df_clean.loc[train_mask, data_used], df_clean.loc[train_mask, 'storm_warning']
X_test, y_test = df_clean.loc[test_mask, data_used], df_clean.loc[test_mask, 'storm_warning']

print(f"Training hours (2022-2023): {len(X_train)} | Testing hours (2024): {len(X_test)}")

# ---------------------------------------------------------
# 4. TRAIN XGBOOST MODEL (High-Recall Storm Sensitivity)
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

print("\n--- 2024 Full-Year Classification Report ---")
print(classification_report(y_test, predictions, zero_division=0))

# Check accuracy at 65% threshold
evacuate_pred = (probs >= 0.65).astype(int)
print("\n--- Tier 2 Evacuation Alert Report (Threshold >= 0.65) ---")
print(classification_report(y_test, evacuate_pred, zero_division=0))


model.save_model("rhizocoast_model.json")
print("Model saved successfully as 'rhizocoast_model.json'")

# ---------------------------------------------------------
# 5b. Event Catch Rate test & 0.5 Threshold test
# ---------------------------------------------------------
evac_50 = (probs >= 0.5).astype(int)
print("\n--- Tier 2 Evacuation Alert Report (Threshold >= 0.5) ---")
print(classification_report(y_test, evac_50, zero_division=0))

# Group consecutive storms into separate events
test_df = df_clean.loc[test_mask].copy()
test_df['prob'] = probs
test_df['event_id'] = (test_df['storm_warning'].diff() == 1).cumsum()

storm_events = test_df[test_df['storm_warning'] == 1].groupby('event_id').agg(
    start_time=('timestamp', 'min'),
    duration_hours=('storm_warning', 'count'),
    max_prob=('prob', 'max')
)

caught_evac = (storm_events['max_prob'] >= 0.5).sum()
caught_watch = ((storm_events['max_prob'] >= 0.15)).sum()
total_events = len(storm_events)

print(f"Total Distinct Storm Events in 2024: {total_events}")
print(f"Total Caught Evacuation-Level Storms (>=0.5): {caught_evac} | Catch Rate: {caught_evac / total_events:.2%}")
print(f"Total Caught Watch-Level Storms (>=0.15): {caught_watch} | Catch Rate: {caught_watch / total_events:.2%}")

# ---------------------------------------------------------
# 6. SIMULATE LIVE SENSOR INFERENCE
# ---------------------------------------------------------
test_live_reading = pd.DataFrame([{
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

storm_probability = model.predict_proba(test_live_reading)[0][1]
print(f"\nSimulated Live Storm Probability: {storm_probability * 100:.1f}%")

if storm_probability >= 0.75:
    print("ALERT STATUS: [LEVEL 2 WARNING] Evacuate / Notify Coastal Authorities!")
elif storm_probability >= 0.15:
    print("ALERT STATUS: [LEVEL 1 WATCH] Abnormal coastal conditions detected.")
else:
    print("ALERT STATUS: [NORMAL] Seawall sensors nominal.")
