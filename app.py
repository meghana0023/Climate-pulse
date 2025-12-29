import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from predict import predict_flood_drought, VALID_STATES

app = Flask(__name__)

@app.route("/")
def home():
    # Sends the list of states to the dropdown in the UI
    return render_template("index.html", states=VALID_STATES)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        state = data.get("state")
        year = int(data.get("year"))
        month = int(data.get("month"))

        # Run your ML prediction logic
        rain, spi, status = predict_flood_drought(state, year, month)
        
        # Clean the data so it doesn't break the JSON response
        clean_spi = 0.0 if np.isnan(spi) else round(float(spi), 2)
        clean_rain = round(float(rain), 2)

        return jsonify({
            "rainfall": clean_rain,
            "spi": clean_spi,
            "status": status
        })
    except Exception as e:
        # If something goes wrong, send the error message to the UI
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)