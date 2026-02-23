from src.adult_income_ml.logger import logging
from src.adult_income_ml.exceptions import customexception
from src.adult_income_ml.components.data_ingestion import DataIngestion,DataIngestionConfig
import sys

if __name__=='__main__':
    logging.info('the execution has started')
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info('custom exception')
        raise customexception(e,sys)