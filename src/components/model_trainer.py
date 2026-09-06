import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from src.constant import *
from src.logger import logging
from src.exception import CustomException
from src.utils import main_utils
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

class PhishingModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        try:
            transformed_feature = self.preprocessing_object.transform(X)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, X):
        try:
            transformed_feature =  self.preprocessing_object.transform(X)
            return self.trained_model_object.predict_proba(transformed_feature)
        except Exception as e:
            raise CustomException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainerConfig:
    def __init__(self):
        timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        self.artifact_dir = os.path.join(ARTIFACT_DIR, timestamp)
        self.model_trainer_dir = os.path.join(self.artifact_dir, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path = os.path.join(self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.expected_accuracy = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path = MODEL_CONFIG_FILE_PATH

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                  model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.models = {
            "GaussianNB" : GaussianNB(),
            "XGBClassifier" : XGBClassifier(objective = 'binary:logistic'),
            "LogisticRegression" : LogisticRegression()
        }

    def evaluate_models(self, X_train, y_train, X_test, y_test, models: dict) -> dict:
        try:
            report = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report[model_name] = accuracy
            return report
        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self, X_train, y_train, X_test, y_test):
        try:
            model_report : dict = self.evaluate_models(
                X_train= X_train, y_train= y_train,
                X_test= X_test, y_test= y_test,
                models= self.models
            )
            logging.info(f"Model report: {model_report}")
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model_object = self.models[best_model_name]
            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object, best_model_name, X_train, y_train):
        try:
            model_param_grid = main_utils.read_yaml_file(self.model_trainer_config.model_config_file_path
                                                         )["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search = GridSearchCV(
                best_model_object, param_grid= model_param_grid, cv= 5, n_jobs= -1, verbose= 1
            )
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            logging.info(f"Best params for {best_model_name} : {best_params}")
            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = main_utils.load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = main_utils.load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1],
                test_arr[:, :-1], test_arr[:, -1]
            )
            best_model_name, best_model_object, best_model_score = self.get_best_model(X_train, y_train, X_test, y_test)

            best_model = self.finetune_best_model(
                best_model_object= best_model_object,
                best_model_name= best_model_name,
                X_train= X_train, y_train= y_train
            )
            best_model.fit(X_train, y_train)
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            logging.info(f"Best model : {best_model_name}, Train accuracy : {train_accuracy}, Test accuracy : {test_accuracy}")

            if test_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception(f"No model found with accuracy greater than {self.model_trainer_config.expected_accuracy}")

            preprocessing_object = main_utils.load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            phishing_model = PhishingModel(
                preprocessing_object= preprocessing_object,
                trained_model_object= best_model
            )

            main_utils.save_object(self.model_trainer_config.trained_model_file_path, phishing_model)
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path= self.model_trainer_config.trained_model_file_path,
                train_accuracy= train_accuracy,
                test_accuracy= test_accuracy
            )
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

