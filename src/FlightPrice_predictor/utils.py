import pandas as pd
import os
import sys
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from src.FlightPrice_predictor.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error


def labelencoder(df,Columns):
    #df = pd.read_csv("artifacts/data.csv")

    for col in Columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

def save_object(file_path,obj):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)

    with open(file_path,'wb') as file_obj:
        pickle.dump(obj,file_obj)



def evaluate_model(xtrain,ytrain,xtest,ytest,models,params):
    try:
        report = {}

        for model_name,model in models.items():
            param = params[model_name]

            grid = GridSearchCV(model,param,cv=3)
            grid.fit(xtrain,ytrain)
            best_param = grid.best_params_
            model.set_params(**best_param)
            model.fit(xtrain,ytrain)
            pred = model.predict(xtest)
            score = r2_score(ytest,pred)

            report[model_name] = score
            print(report)

            return report
        
    except Exception as ex:
        raise CustomException(ex,sys)
    
def evalute_metries(true,pred):
    r2 = r2_score(true,pred)
    mae = mean_absolute_error(true,pred)
    mse = mean_squared_error(true,pred)
    rmse = root_mean_squared_error(true,pred)
    return r2,mae,rmse

from sklearn.metrics import r2_score

def adjusted_r2(true,pred,x):
    """
    Calculate Adjusted R^2 score
    
    Parameters:
    ytrue : array-like, true target values
    ypred : array-like, predicted target values
    x     : array-like or DataFrame, features used in the model
    
    Returns:
    adj_r2 : float, adjusted R^2 score
    """
    n = len(true)        # number of samples
    p = x.shape[1]         # number of features
    r2 = r2_score(true,pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2
