import os
from dataclasses import dataclass
from src.constant import *

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path : str
    transformed_train_file_path :str
    transformed_test_file_path : str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_accuracy: float
    test_accuracy: float

@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = PREDICTION_OUTPUT_DIR_NAME
    prediction_file_name: str = PREDICTION_FILE_NAME
    prediction_file_path: str = os.path.join(PREDICTION_OUTPUT_DIR_NAME, PREDICTION_FILE_NAME)
    