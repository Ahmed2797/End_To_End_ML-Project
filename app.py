import sys
import logging
from src.FlightPrice_predictor.logger import logging 
from src.FlightPrice_predictor.exception import CustomException
from src.FlightPrice_predictor.components.data_ingestion import Data_Ingestion
from src.FlightPrice_predictor.components.data_transformation import DataTransform



if __name__== '__main__':
    logging.info('logger.py Excuation Started')
    try:
        dt_ingestion = Data_Ingestion()
        train_data_path,test_data_path = dt_ingestion.init_data_ingestion()

        dt_preprocessor = DataTransform()
        train_arr,test_arr = dt_preprocessor.init_data_transformation(train_data_path,test_data_path)

    except Exception as e:
        raise CustomException(e, sys)
