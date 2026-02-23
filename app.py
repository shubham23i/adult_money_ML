from src.adult_income_ml.logger import logging
from src.adult_income_ml.exceptions import customexception
import sys

if __name__=='__main__':
    logging.info('the execution has started')
    try:
        a=1/0
    except Exception as e:
        logging.info('custom exception')
        raise customexception(e,sys)