import pandas as pd
import numpy as np

years = [2022, 2023, 2024]
all_years_data = []

for year in years:
    print(f"Downloading Key West data for {year}...")
    
    # 1. NDBC Meteorological Data
    met_url = f"https://www.ndbc.noaa.gov/data/historical/stdmet/kywf1h{year}.txt.gz"
    df_met = pd.read_csv(met_url, sep=r'\s+', skiprows=[1], compression='gzip')
    
    # Handle '#YY' vs 'YYYY' column header differences across NOAA years
    year_col = '#YY' if '#YY' in df_met.columns else 'YYYY'
    df_met['timestamp'] = pd.to_datetime(
        df_met[[year_col, 'MM', 'DD', 'hh', 'mm']].rename(
            columns={year_col: 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}
        )
    )
    df_met['timestamp'] = df_met['timestamp'].dt.round('h')
    df_met = df_met.replace([99.0, 999, 999.0, 9999.0], np.nan)
    
    pres_col = 'PRES' if 'PRES' in df_met.columns else 'BAR'
    df_met = df_met[['timestamp', pres_col, 'WSPD', 'GST', 'WDIR']].rename(columns={pres_col: 'PRES'})
    df_met = df_met.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    # 2. NOAA CO-OPS Hourly Tide Levels
    tide_url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"begin_date={year}0101&end_date={year}1231&station=8724580"
        f"&product=hourly_height&datum=MLLW&units=english&time_zone=gmt&format=csv"
    )
    df_tide = pd.read_csv(tide_url)
    df_tide.columns = df_tide.columns.str.strip()
    df_tide['timestamp'] = pd.to_datetime(df_tide['Date Time'])
    df_tide['TIDE'] = pd.to_numeric(df_tide['Water Level'], errors='coerce')
    df_tide = df_tide[['timestamp', 'TIDE']].drop_duplicates(subset=['timestamp'])
    
    # 3. Merge Year
    df_merged = pd.merge(df_met, df_tide, on='timestamp', how='inner')
    all_years_data.append(df_merged)

# Combine all 3 years chronologically
df_full = pd.concat(all_years_data, ignore_index=True)
df_full = df_full.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

# Interpolate short sensor dropouts (up to 3 hours) and drop remaining NaNs
cols = ['PRES', 'WSPD', 'GST', 'WDIR', 'TIDE']
df_full[cols] = df_full[cols].interpolate(limit=3)
df_clean = df_full.dropna().reset_index(drop=True)

# 4. Auto-Label Storm Warnings (including 6-hour early warning lead time)
is_severe = (
    (df_clean['PRES'] < 1008.0) | 
    (df_clean['WSPD'] > 14.0) | 
    (df_clean['TIDE'] > 3.2)
)

df_clean['storm_warning'] = (
    is_severe | 
    is_severe.shift(-1).fillna(False) | 
    is_severe.shift(-3).fillna(False) | 
    is_severe.shift(-6).fillna(False)
).astype(int)

df_clean.to_csv("key_west_2022_2024.csv", index=False)
print(f"\nDone! Saved {len(df_clean)} total rows (2022–2024) to 'key_west_2022_2024.csv'.")
print(f"Total storm warning hours: {df_clean['storm_warning'].sum()}")
