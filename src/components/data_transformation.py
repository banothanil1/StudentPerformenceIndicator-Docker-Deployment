import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    DataPreprocesserpath : str = os.path.join('Artifact','preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        self.DataConfigPreprocesserpath = DataTransformationConfig()
    def get_data_transformer_object(self,train_df_path):
        '''this method returns the column transformer processor of standardization and encoding'''
        try:
            train_df = pd.read_csv(train_df_path)
            
            numerical_columns = [feature for feature in train_df.columns if feature != 'math_score' and train_df[feature].dtypes != 'O']
            
            categorical_columns = [feature for feature in train_df.columns if train_df[feature].dtypes == 'O']

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            preprocesser = ColumnTransformer([
                ("numerical_pipeline",numerical_pipeline,numerical_columns),
                ("categories_pipeline",categorical_pipeline,categorical_columns)
            ])

            logging.info('standardization and imputation completed for numerical features.')
            logging.info('onehotencoding and imputation and standardization completed for numerical features.')

            return preprocesser
        
        except Exception as e:
            logging.info('Exception has been raised while returing the column transeformer preprocess.')
            raise CustomException(e,sys)

    def intiate_data_transeformation(self,train_data_path,test_data_path):
        '''this function preprocess the object returned by get column transeformer function '''
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Read train df and test df completed')
            train_data_independent = train_df.drop('math_score',axis=1)
            train_data_target = train_df['math_score']
            test_data_independent = test_df.drop('math_score',axis=1)
            test_data_target = test_df['math_score']

            logging.info('Obtaining column transformer preprocessing object')
            train_data_preprocessor = self.get_data_transformer_object(train_data_path)
            test_data_preprocessor = self.get_data_transformer_object(train_data_path)

            train_processed_data = train_data_preprocessor.fit_transform(train_data_independent)
            test_processed_data = train_data_preprocessor.transform(test_data_independent)

            logging.info('Train data transformation completed')
            logging.info('Test data transformation completed')

            save_object(filePath = self.DataConfigPreprocesserpath.DataPreprocesserpath, preProcesser = train_data_preprocessor)
            logging.info('Preprocessor pkl filed saved to artifacts')

            train_arr = np.c_[
                train_processed_data,np.array(train_data_target)
            ]
            test_arr = np.c_[
                    test_processed_data,np.array(test_data_target)
                        ]
            return (train_arr,test_arr,self.DataConfigPreprocesserpath)

        except Exception as e:
            logging.info('Exception has been raised while intiating the data transformation.')
            raise CustomException(e,sys)
            

