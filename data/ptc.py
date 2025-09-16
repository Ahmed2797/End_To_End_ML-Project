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
from src.FlightPrice_predictor.logger import logging 

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in Python script name {file_name} line no {exc_tb.tb_lineno} error message {str(error)}"
    return error_message
 
class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        #super().__init__(error_message)
        
        self.error_message = error_message_detail(error_message,error_details)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message