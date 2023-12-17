from src.utils.exception import CustomException
from src.utils.logger import logging
from pymongo.mongo_client import MongoClient
from dataclasses import dataclass
from dotenv import load_dotenv
import pandas as pd
import os


@dataclass
class DataIngestionConfig:
    artifacts_dir:str = os.path.join(os.getcwd(), "artifacts")
    raw_data:str = os.path.join("artifacts", "raw_data.csv")
    oil:str = os.path.join("artifacts", "oil.csv")
    stores:str = os.path.join("artifacts", "stores.csv")
    holidays:str = os.path.join("artifacts", "holidays.csv")
    env_file_path:str = os.path.join("secrets.env")

class DataIngestion:
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()
        logging.info(">>> DATA INGESTION STARTED <<<")

    def load_dataset(self):

        """
        This function is responsible for loading dataset from the MongoDB Database
        """
        logging.info("executing load_dataset function")
        try:
            logging.info("performing data extraction")

            load_dotenv(self.dataingestionconfig.env_file_path)
    
            MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
            MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
            mongodb_uri = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@storesales.ba5omuw.mongodb.net/?retryWrites=true&w=majority"
            client = MongoClient(mongodb_uri)

            data = pd.DataFrame(client["StoreSales"]["Dataset"].find())
            oil = pd.DataFrame(client["StoreSales"]["Oil"].find())
            stores = pd.DataFrame(client["StoreSales"]["Stores"].find())
            holidays = pd.DataFrame(client["StoreSales"]["Holidays"].find())
            
            logging.info("data extraction successful")

            data.drop("_id", axis = 1, inplace = True)
            oil.drop("_id", axis = 1, inplace = True)
            stores.drop("_id", axis = 1, inplace = True)
            holidays.drop("_id", axis = 1, inplace = True)

            os.makedirs(self.dataingestionconfig.artifacts_dir, exist_ok = True)
            data.to_csv(self.dataingestionconfig.raw_data)
            oil.to_csv(self.dataingestionconfig.oil)
            stores.to_csv(self.dataingestionconfig.stores)
            holidays.to_csv(self.dataingestionconfig.holidays)

            logging.info("data saved to artifacts")
            logging.info(">>> DATA INGESTION COMPLETE <<<")
        
        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))