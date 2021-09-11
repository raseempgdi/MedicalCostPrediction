from django.shortcuts import redirect, render
from django.http import HttpResponse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import joblib


def index(request):
    
    return render(request,'index.html')

def predict(request):
    name = request.POST.get('full-name')
    age = request.POST.get('age')
    gender = request.POST.get('gender')
    bmi = findBmi(float(request.POST.get('height')),float(request.POST.get('weight')))
    children = request.POST.get('children')
    smoker = request.POST.get('smoker')
    # new_model=joblib.load("models/tree_model_96_medical_cost.model")
    new_model = getModel()
    cost = str(new_model.predict([[age,gender,bmi,children,smoker]]).astype(int)).lstrip('[').rstrip(']')

    return render(request,'index.html',{'cost' : cost, 'name' : name, 'bmi' : bmi})


def findBmi(height,weight):
   bmi = round(weight / (height/100)**2,2)

   return bmi

def getModel():
    df = pd.read_csv('Dataset/medical_cost.csv')
    df = df.drop(['region','Unnamed: 0'],axis=1)
    le_data = LabelEncoder()
    sex_l = le_data.fit_transform(df['sex'])
    smoker_l = le_data.fit_transform(df['smoker'])
    df['sex'] = sex_l
    df['smoker'] = smoker_l
    x = df.drop('charges',axis=1)
    y = df['charges']
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=0)
    model = tree.DecisionTreeRegressor(min_samples_leaf=6)
    model = model.fit(xtrain,ytrain)
    
    return model

    

