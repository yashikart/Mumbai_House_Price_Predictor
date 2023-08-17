from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=joblib.load(open('mumbai_House_Price_Predictor.pkl','rb'))
house=pd.read_csv('cleaned_mumbai_house.csv')

@app.route('/',methods=['GET','POST'])
def index():
    area=sorted(house['Area'].unique())
    location=sorted(house['Location'].unique())
    No_of_Bedroom=sorted(house['No. of Bedrooms'].unique())
    Gymnasium = sorted(house['Gymnasium'].unique())
    Car_Parking = sorted(house['Car Parking'].unique())
    Gardens = sorted(house['Landscaped Gardens'].unique())
    Swimming_Pool = sorted(house['Swimming Pool'].unique())

    location.insert(0,'Select Location')
    return render_template('index.html',area=area,location=location,No_of_Bedroom=No_of_Bedroom,Gymnasium=Gymnasium,Car_Parking=Car_Parking,
                           Gardens=Gardens,Swimming_Pool=Swimming_Pool)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    location=request.form.get('location')
    area = request.form.get('area')
    No_of_Bedroom = request.form.get('No_of_Bedroom')
    Gymnasium = request.form.get('Gymnasium')

    Car_Parking=request.form.get('Car_Parking')
    Gardens=request.form.get('Gardens')
    Swimming_Pool=request.form.get('Swimming_Pool')


    prediction = model.predict(pd.DataFrame(columns=['Area','Location','No. of Bedrooms','Gymnasium','Car Parking','Landscaped Gardens','Swimming Pool'],
                              data=np.array([area,location,No_of_Bedroom,Gymnasium,Car_Parking,Gardens,Swimming_Pool]).reshape(1, 7)))



    return str(abs(np.round(prediction[0])))



if __name__=='__main__':
    app.run()