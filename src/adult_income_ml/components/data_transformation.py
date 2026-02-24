import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.adult_income_ml.utils import save_object
from src.adult_income_ml.exceptions import customexception
from src.adult_income_ml.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

            cat_cols = ['workclass', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'native_country']
            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ("ohencoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            logging.info(f"categorical columns:{cat_cols}")
            logging.info(f"numerical columns:{num_cols}")

            preprocessor=ColumnTransformer([
                ("num pipeline",num_pipeline,num_cols),
                ("cat pipeline",cat_pipeline,cat_cols)
            ])
            return preprocessor
        except Exception as e:
            raise customexception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            print("Train DF shape:", train_df.shape)
            print("Test DF shape:", test_df.shape)
            logging.info('reading the train and test files')
            preprocessing_obj=self.get_data_transformation_obj()
            target_column_name="income"
            numerical_column=['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info("applying preprocessing on train and test dataset")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise customexception(e,sys)