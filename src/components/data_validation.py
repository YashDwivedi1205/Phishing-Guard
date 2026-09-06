import os
import sys
import pandas as pd
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from datetime import datetime
from src.exception import CustomException
from src.logger import logging
from src.utils import main_utils
from src.constant import *

class DataValidationConfig:
    def __init__(self):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir = os.path.join(ARTIFACT_DIR, timestamp)
        self.data_validation_dir = os.path.join(self.artifact_dir, DATA_VALIDATION_DIR_NAME)
        self.valid_status_file_path = os.path.join(self.data_validation_dir, DATA_VALIDATION_STATUS_FILE)

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = DataValidationConfig()
        self._schema_config = main_utils.read_yaml_file(SCHEMA_FILE_PATH)

    def validate_number_of_columns(self, dataframe) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e, sys)

    def is_columns_exist(self, dataframe) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns
            missing_columns = []
            for col in numerical_columns:
                if col not in dataframe_columns:
                    missing_columns.append(col)

            if len(missing_columns) > 0:
                    logging.info(f"Missing columns in dataframe: {missing_columns}")
                    return False
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Start Data validation")

            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            validate_column_number_train = self.validate_number_of_columns(train_df)
            validate_column_number_test = self.validate_number_of_columns(test_df)
            if validate_column_number_train is False:
                validation_error_msg += "Columns are missing in training dataframe."
            if validate_column_number_test is False:
                validation_error_msg += "Columns are missing in testing dataframe."

            check_column_exist_train = self.is_columns_exist(train_df)
            check_column_exist_test = self.is_columns_exist(test_df)

            if check_column_exist_train is False:
                validation_error_msg += "All required columns are not present in the training dataframe."
            if check_column_exist_test is False:
                validation_error_msg += "All required columns are not present in the testing dataframe."

            validation_status = len(validation_error_msg) == 0
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            with open(self.data_validation_config.valid_status_file_path, "w") as f:
                f.write(f"Validation Status : {validation_status}")

            data_validation_artifact = DataValidationArtifact(
                validation_status= validation_status,
                message= validation_error_msg
            )
            logging.info(f"Data validation artifact : {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    data_ingestion = DataIngestion()
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
