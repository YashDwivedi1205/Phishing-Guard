import sys
import os
import shutil
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils.main_utils import upload_model_to_huggingface
from src.constant import HUGGINGFACE_REPO_ID, HUGGINGFACE_MODELL_FILENAME

class TrainPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation = DataValidation(data_ingestion_artifact= data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_validation_artifact, data_ingestion_artifact):
        try:
            data_transformation_config = DataTransformationConfig()
            data_transformation = DataTransformation(
                data_validation_artifact= data_validation_artifact,
                data_ingestion_artifact= data_ingestion_artifact,
                data_transformation_config= data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, data_transformation_artifact):
        try:
            model_trainer_config = ModelTrainerConfig()
            model_trainer = ModelTrainer(
                data_transformation_artifact= data_transformation_artifact,
                model_trainer_config= model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact, data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            # saving the final model locally
            os.makedirs("saved_models", exist_ok = True)
            shutil.copy(model_trainer_artifact.trained_model_file_path, os.path.join("saved_models", "model.pkl"))
            upload_model_to_huggingface(
                local_model_path= os.path.join("saved_models", "model.pkl"),
                repo_id= HUGGINGFACE_REPO_ID,
                filename= HUGGINGFACE_MODELL_FILENAME
            )
            logging.info("Model copies to saved models for deployment")
            
            logging.info("Training Pipeline completed successfully")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

        

if __name__ == "__main__":
    pipeline = TrainPipeline()
    result = pipeline.run_pipeline()
    print(result)