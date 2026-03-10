import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
print(preprocessor.feature_names_in_)
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])

def predict():

    input_data = {
        "age": [float(request.form["age"])],
        "workclass": [request.form["workclass"]],
        "fnlwgt": [float(request.form["fnlwgt"])],
        "education": [request.form["education"]],
        "education_num": [float(request.form["education_num"])],
        "marital_status": [request.form["marital_status"]],
        "occupation": [request.form["occupation"]],
        "relationship": [request.form["relationship"]],
        "race": [request.form["race"]],
        "sex": [request.form["sex"]],
        "capital_gain": [float(request.form["capital_gain"])],
        "capital_loss": [float(request.form["capital_loss"])],
        "hours_per_week": [float(request.form["hours_per_week"])],
        "native_country": [request.form["native_country"]]
    }

    df = pd.DataFrame(input_data)

    data = preprocessor.transform(df)

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = ">50K Income"
    else:
        result = "<=50K Income"

    return render_template(
        "index.html",
        prediction_text=f"Predicted Income: {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)