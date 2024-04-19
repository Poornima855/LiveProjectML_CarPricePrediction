from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipelines.pred_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods = ['POST', "GET"])
def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            Car_Name = request.form.get('Car_Name'),
            Selling_Price = float(request.form.get('Selling_Price')),
            Kms_Driven = int(request.form.get("Kms_Driven")), 
            Owner= int(request.form.get("Owner")), 
            Year = int(request.form.get("Year")),
            Fuel_Type = request.form.get("Fuel_Type"), 
            Seller_Type = request.form.get("Seller_Type"), 
            Transmission = request.form.get("Transmission")
        )
    new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    results = round(pred[0],2)

    return render_template("results.html", final_result = results)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", debug=True)
    except Exception as e:
        print("An exception occurred:", e)


#http://127.0.0.1:5000/ in browser