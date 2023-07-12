from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle as pkl


app = Flask(__name__)



@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/car-price-prediction")
def carpriceprediction():    
    dataset = pd.read_csv("cleaned_dataset.csv")
    companies = sorted(dataset["company"].unique())
    names = sorted(dataset["name"].unique())

    return render_template("carpriceprediction.html", companies = companies, names = names)

@app.route("/car-price-prediction-result")
def carpricepredictionresult():
    company = request.args.get("company")
    name = request.args.get("name")
    year = request.args.get("year")
    kms_driven = request.args.get("kms_driven")
    fuel_type = request.args.get("fuel_type")

    model = pkl.load(open('LinearRegresionModel.pkl','rb'))

    result = model.predict(pd.DataFrame([[name,company, year,kms_driven,fuel_type]],columns=["name","company","year","kms_driven","fuel_type"]))

    return render_template("carpricepredictionresult.html", company = company, name = name, year = year, kms_driven = kms_driven, fuel_type = fuel_type, result = int(result[0][0]))