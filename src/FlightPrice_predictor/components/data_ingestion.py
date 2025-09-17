import os 
import sys 
import pandas as pd 
from src.FlightPrice_predictor.logger import logging 
from src.FlightPrice_predictor.exception import CustomException 
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split

@dataclass 
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw.csv")

class Data_Ingestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def init_data_ingestion(self):
        try:
            logging.info('Data Import')
            df = pd.read_csv(r"data/Flight_clean.csv")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Import Succes')
            print(df.head())

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as ex:
            raise CustomException(ex,sys)
        