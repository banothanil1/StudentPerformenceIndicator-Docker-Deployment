import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from src.utils import evaluate_model
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainingConfig:
    ModelTrainingConfigPath = os.path.join('Artifact','model.pkl')

class ModelTraining:
    def __init__(self):
        self.trained_model_path = ModelTrainingConfig()

    def model_train(self, train_arr, test_arr):
        '''this model is used to train the model using the train data and test data'''
        try:
            models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Ridge": Ridge(),
                    "Lasso": Lasso(),
                    "Elastic Net": ElasticNet(),
                    "SVR": SVR(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "AdaBoost": AdaBoostRegressor(),
                    "KNN": KNeighborsRegressor(),
                    "XGBoost": XGBRegressor(),
                    "CatBoost": CatBoostRegressor()
                }
            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            params = {
                "Linear Regression" : {
                    'fit_intercept': [True, False],
                    'positive': [True, False]},
                "Decision Tree" : {
                    'max_depth': [3, 5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': [None, 'sqrt', 'log2']
                } ,
                "Ridge" : {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                "Lasso" : {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [1000, 5000, 10000]
                },
                "Elastic Net" : {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                "SVR" : {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                "Random Forest" : {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [1.0, 'sqrt', 'log2']
                },
                "Gradient Boosting" : {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [2, 3, 5],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "AdaBoost" : {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 10, 15, 20],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "XGBoost" : {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                "CatBoost": {
                    'iterations': [300, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 10]
                }
            }

            model_report = evaluate_model(X_train,y_train,X_test,y_test,models,params = params)

            best_score = max(list(model_report.values()))

            if best_score <= 0.6:
                raise CustomException("No Best Model Found")
                    
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

            best_model_name = models[best_model_name]
            
            logging.info(f"Best Model Name is {best_model_name} and best score is {best_score}")
            
            save_object(filePath= self.trained_model_path.ModelTrainingConfigPath, preProcesser= best_model_name)
            
            logging.info(f'Model converted to pickle file and saved to artifact and path is{self.trained_model_path.ModelTrainingConfigPath}')

            return [best_model_name,best_score]
            
        except Exception as e:
            raise CustomException(e,sys)