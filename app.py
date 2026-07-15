from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        pm25 = float(request.form["pm25"])
        pm10 = float(request.form["pm10"])
        no = float(request.form["no"])
        no2 = float(request.form["no2"])
        nox = float(request.form["nox"])
        nh3 = float(request.form["nh3"])
        co = float(request.form["co"])
        so2 = float(request.form["so2"])
        o3 = float(request.form["o3"])
        features = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3]])

        
        prediction = model.predict(features)
        aqi = round(prediction[0], 2)
        if aqi <= 50:
            category = "🟢 Good"
        elif aqi <= 100:
            category = "🟡 Satisfactory"
        elif aqi <= 200:
            category = "🟠 Moderate"
        elif aqi <= 300:
            category = "🔴 Poor"
        elif aqi <= 400:
            category = "🟣 Very Poor"
        else:
            category = "⚫ Severe"

        return render_template(
            "index.html",
            prediction_text=f"Predicted AQI: {aqi}",
            category=category
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )
if __name__ == "__main__":
    app.run(debug=True)