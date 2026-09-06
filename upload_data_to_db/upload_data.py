import pandas as pd
import json
import sys
sys.path.append(".") #for importing all src module
from src.configuration.mongo_db_connections import MongoDBClient
from src.constant import DATABASE_NAME, COLLECTION_NAME

#CSV file path
FILE_PATH = "upload_data_to_db/phising_08012020_120000.csv"

if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    #Converting dataframe into dictionary
    df.reset_index(drop=True, inplace=True)
    json_records = list(json.loads(df.to_json(orient="records")))

    mongo_client = MongoDBClient()
    mongo_client.client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)