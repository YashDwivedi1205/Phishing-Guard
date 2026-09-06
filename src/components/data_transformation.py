import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
from src.logger import logging
from src.exception import CustomException
from src.utils import main_utils
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact, DataTransformationArtifact
from src.constant import *

class DataTransformationConfig:
    def __init__(self):
        timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        self.artifact_dir = os.path.join(ARTIFACT_DIR, timestamp)
        self.data_transformation_dir = os.path.join(self.artifact_dir ,DATA_TRANSFORMATION_DIR_NAME)
        self.data_transformed_data_dir = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
        self.data_transformation_object_dir = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)
        self.transformed_train_file_path = os.path.join(self.data_transformed_data_dir, TRANSFORMED_TRAIN_FILE_NAME)
        self.transformed_test_file_path = os.path.join(self.data_transformed_data_dir, TRANSFORMED_TEST_FILE_NAME)
        self.transformed_object_file_path = os.path.join(self.data_transformation_object_dir, PREPROCESSING_OBJECT_FILE_NAME)

class DataTransformation:
    def __init__(self, data_validation_artifact : DataValidationArtifact,
                 data_ingestion_artifact:DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        try:
            imputer = SimpleImputer(strategy="most_frequent")
            preprocessor = Pipeline(steps=[("imputer", imputer)])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                preprocessor = self.get_data_transformer_object()

                train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
                test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
                target_feature_train_df = train_df[TARGET_COLUMN]
                target_feature_train_df = target_feature_train_df.replace(-1, 0)

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
                target_feature_test_df = test_df[TARGET_COLUMN]
                target_feature_test_df = target_feature_test_df.replace(-1, 0)

                preprocessor_object = preprocessor.fit(input_feature_train_df)
                transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
                transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

                train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
                test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

                main_utils.save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
                main_utils.save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
                main_utils.save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path= self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path= self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation

    data_ingestion = DataIngestion()
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact = data_validation.initiate_data_validation()

    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(
        data_validation_artifact=data_validation_artifact,
        data_ingestion_artifact=data_ingestion_artifact,
        data_transformation_config=data_transformation_config
    )
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    print(data_transformation_artifact)