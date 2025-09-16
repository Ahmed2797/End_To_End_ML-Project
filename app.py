import sys
import logging
from src.FlightPrice_predictor.logger import logging 
from src.FlightPrice_predictor.exception import CustomException
from src.FlightPrice_predictor.components.data_ingestion import Data_Ingestion



if __name__== '__main__':
    logging.info('logger.py Excuation Started')
    try:
        dt_ingestion = Data_Ingestion()
        train_data_path,test_data_path = dt_ingestion.init_data_ingestion()

    except Exception as e:
        raise CustomException(e, sys)
