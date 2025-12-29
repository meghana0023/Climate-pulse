import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from functools import wraps
from predict import predict_flood_drought, VALID_STATES

# 1. Load the variables from .env
load_dotenv()
SECRET_KEY = os.getenv("CLIMATE_API_KEY")

app = Flask(__name__)

# 2. Security Wrapper Function
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Checks if 'X-API-KEY' header matches our .env file
        user_provided_key = request.headers.get('X-API-KEY')
        if user_provided_key and user_provided_key == SECRET_KEY:
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized Access"}), 401
    return decorated

@app.route("/")
def home():
    return render_template("index.html", states=VALID_STATES)

@app.route("/api/predict", methods=["POST"])
@require_api_key  # Apply the security check
def api_predict():
    try:
        data = request.json
        state = data.get("state")
        year = int(data.get("year"))
        month = int(data.get("month"))

        # Passing all 3 parameters to your ML logic
        rain, spi, status = predict_flood_drought(state, year, month)
        
        # Handle NaN for JSON response
        clean_spi = 0.0 if np.isnan(spi) else round(float(spi), 2)

        return jsonify({
            "rainfall": round(float(rain), 2),
            "spi": clean_spi,
            "status": status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # threaded=True allows the web page to load while models are processing
    app.run(debug=True, port=5000, threaded=True)