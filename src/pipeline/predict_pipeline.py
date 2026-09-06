import os
import sys
import pandas as pd
from flask import request

from src.exception import CustomException 
from src.logger import logging
from src.constant import *
from src.utils import main_utils
from src.entity.artifact_entity import PredictionFileDetail

class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.prediction_file_detail = PredictionFileDetail()

    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = PREDICTION_INPUT_DIR_NAME
            os.makedirs(pred_file_input_dir, exist_ok= True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            input_csv_file.save(pred_file_path)

            return pred_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self):
        try:
            model_path = SAVED_MODEL_FILE_PATH
            if not os.path.exists(model_path):
                logging.info("Local model not found.Downloading from Hugging Face.")
                model_path = main_utils.download_model_from_huggingface(
                    repo_id= HUGGINGFACE_REPO_ID,
                    filename= HUGGINGFACE_MODELL_FILENAME,
                    local_dir= "saved_models"
                )
            return main_utils.load_object(file_path= model_path)
        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, features):
        try:
            model = self.get_model()
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            prediction_column_name = TARGET_COLUMN
            input_dataframe = pd.read_csv(input_dataframe_path)

            model = self.get_model()
            predictions = model.predict(input_dataframe)
            probabilities = model.predict_proba(input_dataframe)

            confidence = [round(max(prob) * 100, 1) for prob in probabilities]

            input_dataframe[prediction_column_name] =  predictions
            target_column_mapping = {0: 'phishing', 1: 'safe'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            input_dataframe['Confidence'] = confidence

            os.makedirs(self.prediction_file_detail.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index= False)
            logging.info("Predictions completed")
            return input_dataframe

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            result_df = self.get_predicted_dataframe(input_csv_path)
            return self.prediction_file_detail, result_df
        except Exception as e:
            raise CustomException(e, sys)