import pandas as pd
import numpy as np
import sys

from src.configuration.mongo_db_connections import MongoDBClient
from src.constant import DATABASE_NAME
from src.exception import CustomException

class PhishingData:
    """ This class exports data from MongoDB and prepares Dataframe."""
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient()
        except Exception as e:
            raise CustomException(e, sys)

    def export_collection_as_dataframe(self, collection_name, database_name=None):
        try:
            if database_name is None:
                database_name = DATABASE_NAME
            collection = self.mongo_client.client[database_name][collection_name]
            df = pd.DataFrame(collection.find())
            if "_id" in df.columns:
                    df.drop(columns=["_id"],  inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    from src.constant import COLLECTION_NAME

    phishing_data = PhishingData()
    df = phishing_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)
    print(df.shape)
    print(df.head())