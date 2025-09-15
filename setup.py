from setuptools import setup,find_packages 
from typing import List 

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [reg.strip() for reg in requirements if reg.strip()]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements 

setup(
    name= 'FlightPrice_predictor',
    version= '0.0.1',
    author= 'Ahmed',
    author_email= 'tanvirahmed754575@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')
)