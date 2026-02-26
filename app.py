from src.adult_income_ml.logger import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from src.adult_income_ml.exceptions import customexception
from src.adult_income_ml.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.adult_income_ml.components.data_transformation import DataTransformation,DataTransformationConfig
from src.adult_income_ml.components.model_trainer import ModelTrainerConfig,ModelTrainer
import sys

if __name__=='__main__':
    logging.info('the execution has started')
    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    except Exception as e:
        logging.info('custom exception')
        raise customexception(e,sys)