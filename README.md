# RhizoCoast AI — Predictive Storm Warning Model

Machine learning pipeline for **RhizoCoast AI**, an artificial mangrove-inspired biochar seawall and coastal early-warning system. This model analyzes real-time barometric pressure, wind vectors, and tidal surge rates to predict incoming extreme weather up to 6 hours in advance.

## Project Structure

* `download_data.py` — Fetches and merges multi-year historical weather (`KYWF1`) and hourly tide data (`8724580`) from NOAA for Key West, FL (2022–2024).
* `main.py` — Engineers atmospheric trend features, trains an XGBoost classifier on 2022–2023 data, evaluates performance on 2024 data, and exports the trained model.
* `rhizocoast_model.json` — Saved XGBoost model weights ready for hardware or web dashboard inference.
* `requirements.txt` — Python dependencies.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
