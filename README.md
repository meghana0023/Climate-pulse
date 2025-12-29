# Climate-Driven Flood & Drought Prediction System

## Features
- LSTM climate prediction per state
- CNN-LSTM rainfall forecasting
- SPI-3 based flood/drought classification
- Flask web interface

## Run
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
python app.py

# 🌧 Climate Pulse API
A Rainfall and Flood/Drought prediction system using CNN-LSTM Deep Learning.

## Setup Instructions
1. Clone the repo: `git clone https://github.com/meghana0023/climate-pulse.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root folder and add:
   `CLIMATE_API_KEY=your_chosen_key`
4. Run the app: `python app.py`

## Features
- **Symmetrical UI**: Dual-panel input and output reporting.
- **Deep Learning**: Uses trained models to forecast rainfall based on region and time.
- **Security**: Protected by an API key header.