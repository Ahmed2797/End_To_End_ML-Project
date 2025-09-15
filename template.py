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