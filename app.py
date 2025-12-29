from flask import Flask, render_template, request, jsonify
from predict import predict_flood_drought, VALID_STATES
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from functools import wraps
app = Flask(__name__)

# 1. Loading the variables from .env
load_dotenv()
SECRET_KEY = os.getenv("CLIMATE_API_KEY")



@app.route("/")
def home():
    return render_template("index.html", states=VALID_STATES)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Checks if 'X-API-KEY' header matches our .env file
        user_provided_key = request.headers.get('X-API-KEY')
        if user_provided_key and user_provided_key == SECRET_KEY:
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized Access"}), 401
    return decorated

@app.route("/api/predict", methods=["POST"])
@require_api_key  # 3. Apply the security check
def api_predict():
    # Your prediction logic here...
    return jsonify({"status": "Success"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        state = data.get("state")
        year = int(data.get("year"))
        month = int(data.get("month"))

        # Passing all 3 parameters
        rain, spi, status = predict_flood_drought(state, year, month)
        
        # Handle NaN for JSON response
        clean_spi = 0.0 if np.isnan(spi) else round(spi, 2)

        return jsonify({
            "rainfall": round(rain, 2),
            "spi": clean_spi,
            "status": status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # threaded=True allows the web page to load while models are processing
    app.run(debug=True, port=5000, threaded=True)