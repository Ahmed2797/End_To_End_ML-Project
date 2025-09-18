import sys
import logging
from src.FlightPrice_predictor.logger import logging 
from src.FlightPrice_predictor.exception import CustomException
from src.FlightPrice_predictor.components.data_ingestion import Data_Ingestion
from src.FlightPrice_predictor.components.data_transformation import DataTransform
from src.FlightPrice_predictor.components.model_trainer import Model_Trainer



if __name__== '__main__':
    logging.info('logger.py Excuation Started')
    try:
        dt_ingestion = Data_Ingestion()
        train_data_path,test_data_path = dt_ingestion.init_data_ingestion()

        dt_preprocessor = DataTransform()
        train_arr,test_arr,_ = dt_preprocessor.init_data_transformation(train_data_path,test_data_path)
        print('<------------------->')
        print("Train array shape:", train_arr.shape)
        print("Test array shape:", test_arr.shape)
        print("Preprocessor saved at:",_)

        model_train = Model_Trainer()
        print(model_train.init_model_train(train_arr,test_arr))

    except Exception as e:
        raise CustomException(e, sys)
