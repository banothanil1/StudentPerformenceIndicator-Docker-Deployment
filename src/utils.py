import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

def save_object(filePath, preProcesser):
    '''this helper functiion will save the Processor in the provided filePath'''
    try:
        dir_name = os.path.dirname(filePath)
        os.makedirs(dir_name,exist_ok=True)

        with open(filePath,'wb') as fileObj:
            dill.dump(preProcesser,fileObj)

    except CustomException as e:
        raise (e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    '''this helper function is used to train the model and return the model score '''
    try:
        model_report = {}
        for label,model in models.items():

            hyperparameters = params[label]
            grid_processor = GridSearchCV(estimator=model,param_grid=hyperparameters,n_jobs=-1,cv=2,verbose=3,scoring='neg_mean_squared_error')
            grid_processor.fit(X_train,y_train)
            
            logging.info("GridSearch cv performed and fitted the model with best Params..")

            model.set_params(**grid_processor.best_params_)

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            logging.info('Model predicted for Test Data')
            
            model_score = r2_score(y_test,y_pred)
            logging.info('R2 Score Evaluation Completed')
            model_report[label] = model_score

        return model_report
    
    except Exception as e:
        raise CustomException(e,sys)

def load_object (filePath):
    '''this helper function will load the object based on the filePath passed and returns'''
    try:
        with open (filePath,'rb') as obj:
            object = dill.load(obj)
        return object 
    
    except Exception as e:
        raise CustomException(e,sys)