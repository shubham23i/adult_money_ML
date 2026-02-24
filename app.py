from src.adult_income_ml.logger import logging
from src.adult_income_ml.exceptions import customexception
from src.adult_income_ml.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.adult_income_ml.components.data_transformation import DataTransformation,DataTransformationConfig
import sys

if __name__=='__main__':
    logging.info('the execution has started')
    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    except Exception as e:
        logging.info('custom exception')
        raise customexception(e,sys)