import os
import sys
from src.adult_income_ml.logger import logging
from src.adult_income_ml.exceptions import customexception
import pandas as pd
import pymysql
from dotenv import load_dotenv
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
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
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]
            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)
        

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise customexception(e,sys)