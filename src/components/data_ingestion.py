import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.exception import CustomException
from src.logger import logging
from src.data_access.phishing_data import PhishingData
from src.entity.artifact_entity import DataIngestionArtifact
from src.constant import *


class DataIngestionConfig:
    def __init__(self):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir = os.path.join(ARTIFACT_DIR, timestamp)
        self.data_ingestion_dir = os.path.join(self.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, FEATURE_STORE_DIR_NAME, FILE_NAME)
        self.training_file_path = os.path.join(self.data_ingestion_dir, TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.data_ingestion_dir, TEST_FILE_NAME)
        self.train_test_split_ratio = TRAIN_TEST_SPLIT_RATIO

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def export_data_into_feature_store(self):
        try:
            logging.info("Exporting data from MongoDB")
            phishing_data = PhishingData()
            dataframe = phishing_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)
            dataframe = dataframe.drop_duplicates()
            feature_store_file = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_file, exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path ,index=False, header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def split_data_as_train_test(self, dataframe : pd.DataFrame):
        try:
            logging.info("Performing train test split")
            train_set, test_set = train_test_split(dataframe,
                test_size= self.data_ingestion_config.train_test_split_ratio)
            os.makedirs(self.data_ingestion_config.data_ingestion_dir, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index= False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index= False)

            logging.info("Train-Test split completed")

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe)
            logging.info("Data ingestion completed successfully")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path= self.data_ingestion_config.training_file_path,
                test_file_path= self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()