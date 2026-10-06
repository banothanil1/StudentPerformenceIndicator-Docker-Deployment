from flask import Flask,request,render_template
from src.pipelines.predict_pipeline import PredictPipeline
from src.exception import CustomException
import sys

application = Flask(__name__)
app = application

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/predict_datapoint', methods = ['GET','POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home_page.html')
        else :
            gender = request.form.get("gender")
            ethnicity = request.form.get("ethnicity")
            parental_level_of_education = request.form.get("parental_level_of_education")
            lunch = request.form.get("lunch")
            test_preparation_course = request.form.get("test_preparation_course")
            writing_score = float(request.form.get("writing_score"))
            reading_score = float(request.form.get("reading_score"))
            PredictPipelineObj = PredictPipeline(gender,ethnicity,parental_level_of_education,lunch,test_preparation_course,writing_score,reading_score)
            new_data_as_df = PredictPipelineObj.convert_to_dataframe()
            predicted_score = PredictPipelineObj.predict_score(new_data_as_df)
            return render_template('home_page.html',predicted_score=predicted_score)
        
    except Exception as e:
        raise CustomException(e,sys)
    


if __name__ ==  "__main__":
    app.run(host="0.0.0.0")
