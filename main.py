# import the required libraries 
import uvicorn
from fastapi import FastAPI
import pandas as pd 
import pickle
import numpy as np
from pydantic import BaseModel

# create an instance of fastapi

app= FastAPI()

# loading the model 

classifier = pickle.load(open('irismodel.pkl','rb'))

class Inputs(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    
@app.get('/')
def index():
    return {'message':'Hello world'}

@app.post('/predict')
def predict_species(data:Inputs):
    data = data.dict()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    
    prediction = classifier.predict([[sepal_length,sepal_width, petal_length,petal_width]])
    
    return {
        'prediction': prediction[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)


    
    