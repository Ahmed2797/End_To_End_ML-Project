# setup.py

from setuptools import setup,find_packages
from typing import List

HYPHAN_E_DOT = '-e .'

def get_requirement(file_path:str)->List[str]:

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [reg.strip() for reg in requirements if reg.strip()]

        # [reg for ref in requirement if HYPAN_E_DOT in reg] --> '-e .'

        if HYPHAN_E_DOT in requirements:
            requirements.remove(HYPHAN_E_DOT)
        # [reg for ref in requirement if HYPAN_E_DOT not in reg]

    return requirements


setup(
    name= 'FlightPrice_predictor',
    version= '0.0.1',
    author= 'Ahmed',
    author_email= 'tanvirahmed754575@gmail.com',
    packages= find_packages(),
    requires= get_requirement('requirements.txt')
)

'''
        # conda activate env
        # python setup.py install
'''


# template.py

import os 
from pathlib import Path 
import logging 

logging.basicConfig(
    filename='setup',
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

preject_name = 'FlightPrice_predictor'

list_of_file = [
    f'src/{preject_name}/__init__.py',
    f'src/{preject_name}/components/__init__.py',
    f'src/{preject_name}/components/data_ingestion.py',
    f'src/{preject_name}/components/data_transformation.py',
    f'src/{preject_name}/components/model_trainer.py',
    f'src/{preject_name}/pipeline/train_pipeline.py',
    f'src/{preject_name}/components/predict_pipeline.py',
    f'src/{preject_name}/exception.py',
    f'src/{preject_name}/logger.py',
    f'src/{preject_name}/utils.py',
    'app.py',
    'main.py',
    'setup.py',
    'requirements.txt'

]
for filepath in list_of_file:
    filepath = Path(filepath)
    file_dir,file_name = os.path.split(filepath)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f'Creating Directory :{file_dir} ot the {file_name}')
    if (not os.path.exists(filepath) or os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass 
        logging.info(f'Creating {filepath} is empty')
    else:
        logging.info(f'{filepath} already exists') 

'''
        python template.py  # cookic_cutter
'''

# logger.py

import os 
import logging 
from datetime import datetime 

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'logs',log_file)
os.makedirs(log_path,exist_ok=True)

log_file_path = os.path.join(log_path,log_file)

logging.basicConfig(
    filename= log_file_path,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

logging.info("Logging setup completed successfully!")

'''
            from src.FlightPrice_preditor.logger import logging

            if __name__=='__main__':
                logging.info('logger.py Excuation Started')

            ## python app.py
''' 

# exception.py 

import sys 
import logging 


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f'Error occurred in Python script {file_name} at line [{exc_tb.tb_lineno} with error message {str(error)}'
    return error_message
                    
 
class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details)
    def __str__(self):
        return self.error_message
    


# data_ingestion.py 

import os 
import sys 
from dataclasses import dataclass 
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.FlightPrice_predictor.exception import CustomException 
from src.FlightPrice_predictor.logger import logging 

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv') 
    test_data_path:str = os.path.join('artifacts','test.csv') 
    raw_data_path:str = os.path.join('artifacts','raw.csv') 

class Data_ingestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        try:
            
            df = pd.read_csv(r"C:\Users\tanvi\Downloads\Machine Learning\Dataset\airlines_flights_data.csv")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)

            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        
        except Exception as ex:
            raise CustomException(ex,sys)


'''
if __name__ = '__main__':
    dt_ingestion = Data_ingestion()
    train_data_path,test_data_path = dt_ingestion.init_data_ingestion()

            # python app.py

'''

# data_trainsformation.py 

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
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder


@dataclass 
class DataTransformationConfig:
    preprocessor_pkl = os.path.join('artifacts','preprocessor.pkl')

class DataTransform:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transform(self,df: pd.DataFrame):
        try:
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object','category']).columns.tolist()
            #label_encode_cols = df.select_dtypes(include=['object','category']).columns.tolist()

            num_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('one',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, numeric_columns),
            ('cat', cat_pipeline, categorical_columns)
            ])

            return preprocessor 

        except Exception as e:
            raise CustomException(e,sys)
        
    def init_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'price'
            xtrain = train_df.drop(columns=[target_column],axis=1)
            xtest = test_df.drop(columns=[target_column],axis=1)
            ytrain = train_df[target_column]
            ytest = test_df[target_column]

            preprocessor_obj = self.get_data_transform(train_df)

            xtrain_data_arr = preprocessor_obj.fit_transform(xtrain)
            xtest_data_arr = preprocessor_obj.transform(xtest)

            train_arr = np.c_[xtrain_data_arr,np.array(ytrain)]
            test_arr = np.c_[xtest_data_arr,np.array(ytest)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_pkl,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_pkl
                
            )
        
        except Exception as ex:
            raise CustomException(ex,sys)
        
'''
        if __name__== '__main__':
        logging.info('logger.py Excuation Started')
        try:
            dt_ingestion = Data_Ingestion()
            train_data_path,test_data_path = dt_ingestion.init_data_ingestion()

            dt_preprocessor = DataTransform()
            train_arr,test_arr,_ = dt_preprocessor.init_data_transformation(train_data_path,test_data_path)

            # python app.py
'''


# model_trainer.py

import os 
import sys 
import pandas as pd 
import numpy as np
from dataclasses import dataclass 
from src.FlightPrice_predictor.exception import CustomException
from src.FlightPrice_predictor.logger import logging
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.FlightPrice_predictor.utils import *





@dataclass 
class ModelTrainerConfig:
    modeltrainer_pkl = os.path.join('artifacts','best_model.pkl')

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def init_model_train(self,train_arr,test_arr):
        logging.info("<----Model_Traning on the way---->")
        try:
            xtrain = train_arr[:,:-1]
            ytrain = train_arr[:,-1]
            xtest = test_arr[:,:-1]
            ytest = test_arr[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            # models[key] == params[key]

            params = {
                "Decision Tree": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_depth': [3, 5, 10, None],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 5],
                    #'max_features': ['sqrt', 'log2', None]
                },
                "Random Forest": {
                    #'n_estimators': [50, 100, 200, 300],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_depth': [5, 10, None],
                    # 'max_features': ['sqrt', 'log2', None],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    #'n_estimators': [100, 200, 300],
                    #'learning_rate': [0.1, 0.05, 0.01],
                    # 'subsample': [0.6, 0.8, 1.0],
                    # 'max_depth': [3, 5, 7],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 5]
                },
                "Linear Regression": {
                    'fit_intercept': [True, False]
                },

                "XGBRegressor": {
                    #'n_estimators': [100, 200, 300],
                    #'learning_rate': [0.1, 0.05, 0.01],
                    # 'max_depth': [3, 5, 7],
                    # 'subsample': [0.6, 0.8, 1.0],
                    # 'colsample_bytree': [0.6, 0.8, 1.0],
                    # 'gamma': [0, 1, 5]
                },
                "CatBoosting Regressor": {
                    #'iterations': [100, 200, 300],
                    # 'depth': [6, 8, 10],
                    # 'learning_rate': [0.1, 0.05, 0.01],
                    # 'l2_leaf_reg': [1, 3, 5, 7]
                },
                "AdaBoost Regressor": {
                    #'n_estimators': [50, 100, 200],
                    # 'learning_rate': [0.1, 0.5, 1.0],
                    # 'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report  = evaluate_model(xtrain,ytrain,xtest,ytest,models,params)
            best_model_name = max(model_report,key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            best_param = params[best_model_name]

            print('Best_model Name:',best_model_name)
            print("Best_model_score:",best_model_score)
            print('Best params:',best_param)

            if best_model_score < 0.6:
                logging.info('Best model doesnot found')
            
            pred = best_model.predict(xtest)
            R2_score = r2_score(ytest,pred)
            print("R2 Score:",R2_score)

            save_object(
                self.model_trainer_config.modeltrainer_pkl,
                best_model
            )

            return R2_score
        

        except Exception as ex:
            raise CustomException(ex,sys)
'''
if __name__== '__main__':
    logging.info('logger.py Excuation Started')
    try:
        model_train = Model_Trainer()
        print(model_train.init_model_train(train_arr,test_arr))

    except Exception as e:
        raise CustomException(e, sys)

'''





#Air-Flight-Price_Predictor