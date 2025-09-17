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


@dataclass 
class DataTransformationConfig:
    preprocessor_pkl:str = os.path.join('artifacts','preprocessor.pkl')


class DataTransform:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform(self, df: pd.DataFrame):

        '''
        this function is responsible for data transformation
        '''
        try:
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            num_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numeric_columns),
                ('cat', cat_pipeline, categorical_columns)
            ])

            return preprocessor 

        except Exception as e:
            raise CustomException(e, sys)
        
    def init_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_column = 'price'
            xtrain = train_df.drop(columns=[target_column], axis=1)
            xtest = test_df.drop(columns=[target_column], axis=1)
            ytrain = train_df[target_column]
            ytest = test_df[target_column]

            print("Train Columns:", xtrain.columns.tolist())
            print("Test Columns:", xtest.columns.tolist())

            preprocessor_obj = self.get_data_transform(xtrain)
            #print(preprocessor_obj)

            xtrain_data_arr = preprocessor_obj.fit_transform(xtrain)
            xtest_data_arr = preprocessor_obj.transform(xtest)

            ytrain_array = np.array(ytrain).reshape(-1,1)
            ytest_array = np.array(ytest).reshape(-1,1)

            print("xtrain_data_arr shape:", xtrain_data_arr.shape)
            print("xtest_data_arr shape:", xtest_data_arr.shape)
            print("ytrain shape:", ytrain.shape)
            print("ytest shape:", ytest.shape)
            print('ytrain_array.shape:',ytrain_array.shape)
            print('ytest_array.shape:',ytest_array.shape)
          
            # if len(xtrain_data_arr.shape) == 1:
            #     xtrain_data_arr = xtrain_data_arr.reshape(-1, 1)
            # elif xtrain_data_arr.shape[0] == 1:
            #     xtrain_data_arr = xtrain_data_arr.T

            # if len(xtest_data_arr.shape) == 1:
            #     xtest_data_arr = xtest_data_arr.reshape(-1, 1)
            # elif xtest_data_arr.shape[0] == 1:
            #     xtest_data_arr = xtest_data_arr.T

            if hasattr(xtrain_data_arr, 'toarray'):
                xtrain_data_arr = xtrain_data_arr.toarray()
            if hasattr(xtest_data_arr, 'toarray'):
                xtest_data_arr = xtest_data_arr.toarray()
                        
            train_arr = np.c_[xtrain_data_arr,ytrain_array]
            test_arr = np.c_[xtest_data_arr,ytest_array]

            save_object(
                file_path=self.data_transformation_config.preprocessor_pkl,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_pkl
            )
        
        except Exception as ex:
            raise CustomException(ex, sys)


def save_object(file_path, obj):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)

    with open(file_path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)
