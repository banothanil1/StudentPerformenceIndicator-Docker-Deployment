import pandas as pd
import sys
from src.utils import load_object
from src.exception import CustomException
import os

class PredictScore : 
    def __init__(self):
        pass

class PredictPipeline:
    def __init__ (self,gender,ethnicity,parental_level_of_education,lunch,test_preparation_course,writing_score,reading_score):
        self.gender = [gender]
        self.ethnicity = [ethnicity]
        self.parental_level_of_education = [parental_level_of_education]
        self.lunch = [lunch]
        self.test_preparation_course = [test_preparation_course]
        self.writing_score = [writing_score]
        self.reading_score = [reading_score]

    def predict_score(self,new_data):
        '''this function will take the new input data points and applies standardization and predict the score'''
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(BASE_DIR, 'Artifact', 'model.pkl')
            preprocessor_path = os.path.join(BASE_DIR, 'Artifact', 'preprocesser.pkl')
            model = load_object(filePath = model_path)            
            print("MODEL:", model)
            print("MODEL TYPE:", type(model))
            preprocessor = load_object(filePath = preprocessor_path)

            new_data_scaled = preprocessor.transform(new_data)
            predicted = model.predict(new_data_scaled)
            return predicted[0]
        
        except Exception as e:
            raise CustomException(e,sys)

    def convert_to_dataframe(self):
        '''this function converts the raw recieved to Dataframe'''
        try:
            data = {
                "gender" : self.gender,
                "race_ethnicity" : self.ethnicity,
                "parental_level_of_education" : self.parental_level_of_education,
                "lunch" : self.lunch,
                "test_preparation_course" : self.test_preparation_course,
                "writing_score" : self.writing_score,
                "reading_score" : self.reading_score
            }
            new_data = pd.DataFrame(data)
            return new_data
        
        except Exception as e:
            raise CustomException(e,sys)
