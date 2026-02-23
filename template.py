import os
from pathlib import Path 
import logging


logging.basicConfig(level=logging.INFO)
project_name='adult_income_ml'
list_of_files=[
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/exceptions.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/logger.py",
    "app.py",
    "docker.py",
    "requirements.txt",
    "setup.py",    
]

for filepath in list_of_files:
    filepath= Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir!='':
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creting directory :{filedir} for file{filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"creating empty file:{filepath}")
    
    else:
        logging.info(f"{filepath} already existh")