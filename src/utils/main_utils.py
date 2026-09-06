import os
import sys
import yaml
import numpy as np
import dill
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import logging

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file) # safe_load converts contents into python dictionary
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path:str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys)

def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def upload_model_to_huggingface(local_model_path: str, repo_id: str, filename: str) ->None:
    try:
        api = HfApi(token= HUGGINGFACE_TOKEN)
        api.upload_file(
            path_or_fileobj= local_model_path,
            path_in_repo= filename,
            repo_id= repo_id,
            repo_type= "model",
        )
        logging.info(f"Model uploaded to Hugging Face: {repo_id}")

    except Exception as e:
        raise CustomException(e, sys)

def download_model_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    try:
        downloaded_path = hf_hub_download(
            repo_id= repo_id,
            filename= filename,
            local_dir= local_dir,
            token= HUGGINGFACE_TOKEN,
        )
        logging.info(f"Model downloaded from Hugging Face to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        raise CustomException(e, sys)