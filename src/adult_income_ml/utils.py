import os
import sys
from src.adult_income_ml.logger import logging
from src.adult_income_ml.exceptions import customexception
import pandas as pd
import pymysql
from dotenv import load_dotenv
import pickle
load_dotenv()
host=os.getenv('host')
user=os.getenv('user')
password=os.getenv('password')
database=os.getenv('database')
def read_sql_data():
    try:
        logging.info('reading mysql db started')
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logging.info('comleted reading mysql db started')
        df=pd.read_sql_query('select * from adult',mydb)
        return df
    except Exception as e:
        raise customexception(e,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
