import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation 
from sklearn.model_selection import train_test_split
from src.components.model_training import ModelTraining

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("Artifact","train.csv")
    test_data_path: str = os.path.join("Artifact","test.csv")
    raw_data_path: str = os.path.join("Artifact","raw.csv")

class DataIngestion:
    def __init__ (self):
        self.config_paths = DataIngestionConfig()

    def data_ingestion (self):
        logging.info('Entered into data ingestion Method')
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            os.makedirs(os.path.dirname(self.config_paths.raw_data_path),exist_ok=True)
            df.to_csv(self.config_paths.raw_data_path,index=False,header=True)
            logging.info('Raw data path is created and raw.csv has been created.')

            train_data, test_data = train_test_split(df,test_size=0.33,random_state=42)
            logging.info("Train test splitting Performed successfully.")

            train_data.to_csv(self.config_paths.train_data_path,index=False,header=True)
            logging.info("Train Data Created.")

            test_data.to_csv(self.config_paths.test_data_path,index=False,header=True)
            logging.info("Test Data Created")

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.data_ingestion()
    print(obj.config_paths.train_data_path,obj.config_paths.test_data_path)
    data_transform_obj = DataTransformation()
    train_arr, test_arr, _ =  data_transform_obj.intiate_data_transeformation(obj.config_paths.train_data_path,obj.config_paths.test_data_path)
    model_train_obj = ModelTraining()
    best_model_name, best_score = model_train_obj.model_train(train_arr=train_arr,test_arr=test_arr)
    print(best_score)

    
