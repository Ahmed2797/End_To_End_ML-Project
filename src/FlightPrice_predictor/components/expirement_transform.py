import os 
import sys 
import pickle
import numpy as np
import pandas as pd 
from dataclasses import dataclass 
from src.FlightPrice_predictor.exception import CustomException 
from src.FlightPrice_predictor.logger import logging 
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.FlightPrice_predictor.utils import *


@dataclass 
class Data_Transformation_Config:
    labe_expirement_pkl = os.path.join('artifacts','label_expirement.pkl')


class Data_Transform:
    def __init__(self):
        self.data_transformation_config = Data_Transformation_Config()

    def get_data_transform(self, df: pd.DataFrame):

        '''
        this function is responsible for data transformation
        '''
        try:
            #numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns

            data = labelencoder(df,categorical_columns)

            return data



            # num_pipeline = Pipeline(steps=[
            #     ('impute', SimpleImputer(strategy='median')),
            #     ('scaler', StandardScaler())
            # ])

            # cat_pipeline = Pipeline(steps=[
            #     ('impute', SimpleImputer(strategy='most_frequent')),
            #     ('onehot', OneHotEncoder(handle_unknown='ignore')),
            #     ('scaler', StandardScaler(with_mean=False))
            # ])

            # preprocessor = ColumnTransformer(transformers=[
            #     ('num', num_pipeline, numeric_columns),
            #     ('cat', cat_pipeline, categorical_columns)
            # ])

            # return preprocessor 

        except Exception as e:
            raise CustomException(e, sys)
        
    def init_dt_tf(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_column = 'price'
            xtrain = train_df.drop(columns=[target_column], axis=1)
            xtest = test_df.drop(columns=[target_column], axis=1)
            ytrain = train_df[target_column]
            ytest = test_df[target_column]

            xtrain = self.get_data_transform(xtrain)
            xtest = self.get_data_transform(xtest)

            preprocessor_obj = StandardScaler()
            preprocessor_obj.fit(xtrain)
            xtrain_scaled = preprocessor_obj.transform(xtrain)
            xtest_scaled = preprocessor_obj.transform(xtest)


            xtrain_array = pd.DataFrame(xtrain_scaled, columns=xtrain.columns).to_numpy()
            xtest_array = pd.DataFrame(xtest_scaled, columns=xtest.columns).to_numpy()

            import numpy as np

            train_arr = np.concatenate([xtrain_array, ytrain.to_numpy().reshape(-1,1)], axis=1)
            test_arr = np.concatenate([xtest_array, ytest.to_numpy().reshape(-1,1)], axis=1)



            # train_df = pd.concat([xtrain, ytrain], axis=1)
            # train_df = pd.concat([xtest, ytest], axis=1)

            save_object(
                file_path=self.data_transformation_config.labe_expirement_pkl,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.labe_expirement_pkl
            )
        
        except Exception as ex:
            raise CustomException(ex, sys)


def save_object(file_path, obj):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)

    with open(file_path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)
