import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder

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

