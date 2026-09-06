import os
import pymongo
import certifi
from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")

class MongoDBClient:
    def __init__(self):
        if MONGODB_URL is not None:
           self.client = pymongo.MongoClient(MONGODB_URL, tlsCAFile=certifi.where())
        else:
           raise Exception("Environment variable MONGODB_URL is not set")
