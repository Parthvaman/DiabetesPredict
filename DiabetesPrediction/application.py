from flask import Flask,request,app,render_template
from flask import Response
import pandas as pd 
import numpy as np 
import pickle

application=Flask(__name__)
app=application

scaler=pickle.load(open("E:\DiabetesPrediction\Model\standardScalar.pkl","rb"))
model=pickle.load(open("E:\DiabetesPrediction\Model\modelForPrediciton.pkl","rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""
    if request.method=='POST':
        Pregnancies=int(request.form.get("Pregnancies"))
        BloodPressure=float(request.form.get("BloodPressure"))
        SkinThickness=float(request.form.get("SkinThickness"))
        BMI=float(request.form.get("BMI"))
        Insulin=float(request.form.get("Insulin"))
        DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
        Age=int(request.form.get("Age"))

        new_data=scaler.transform([[Pregnancies,BloodPressure,SkinThickness,BMI,Insulin,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)

        if predict[0]==1:
            result='Diabetic'
        else :
            result='Not_Diabetic'
        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')
    

if __name__=="__main__":
    app.run(host="0.0.0.0")