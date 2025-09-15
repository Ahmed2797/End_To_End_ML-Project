import sys
import logging
from src.FlightPrice_predictor.logger import logging 
from src.FlightPrice_predictor.exception import CustomException



if __name__== '__main__':
    logging.info('logger.py Excuation Started')
    try:
        x = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)
    
