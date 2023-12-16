from src.utils.exception import CustomException
from src.utils.logger import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os

@dataclass
class DataTransformationConfig:
    oil:str = os.path.join("artifacts", "oil.csv")
    data:str = os.path.join("artifacts", "raw_data.csv")
    stores:str = os.path.join("artifacts", "stores.csv")
    holidays:str = os.path.join("artifacts", "holidays.csv")
    processed_data:str = os.path.join("artifacts", "processed_data.csv")
    train_data:str = os.path.join("artifacts", "train_data.csv")
    test_data:str = os.path.join("artifacts", "test_data.csv")


class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()

    def transform_data(self):
        logging.info(">>> DATA TRANSFORMATION STARTED <<<")
        try:
            oil = pd.read_csv(self.datatransformationconfig.oil)
            data = pd.read_csv(self.datatransformationconfig.data)
            stores = pd.read_csv(self.datatransformationconfig.stores)
            holidays = pd.read_csv(self.datatransformationconfig.holidays)

            logging.info("handling missing values")

            all_dates = data["date"]
            missing_dates = pd.DataFrame(all_dates[~all_dates.isin(oil.date)].unique())
            missing_dates.rename(columns={0:"date"}, inplace=True)
            oil = pd.concat([oil, missing_dates], axis=0).reset_index(drop=True).sort_values(by="date")
            oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().bfill()

            logging.info("missing values handled successfully")
            logging.info("initialising data integration")

            processed_data = pd.merge(left=data, right=oil, on="date", how="left")
            processed_data = pd.merge(left = processed_data, right = stores, on = "store_nbr", how = "left")
            processed_data.rename(columns={"type":"store_type"}, inplace=True)
            
            holidays = holidays[holidays["transferred"] != True]
            holidays.drop(["transferred", "description"], axis = 1, inplace = True)
            holidays = holidays[holidays["type"] != "Work Day"]
            holidays.drop("type", axis = 1, inplace = True)

            logging.info("removing data inconsistencies")

            national_holidays = holidays[holidays["locale"] == "National"][["date"]]
            regional_holidays = holidays[((holidays["locale"] == "Regional") & (~holidays["date"].isin(national_holidays["date"])))]
            local_holidays = holidays[((holidays["locale"] == "Local") & (~holidays["date"].isin(national_holidays["date"])))]
            regional_holidays = regional_holidays[["date", "locale_name"]].rename(columns={"locale_name":"state"})
            local_holidays = local_holidays[["date", "locale_name"]].rename(columns={"locale_name":"city"})
            national_holidays = national_holidays[~national_holidays.duplicated(keep='first')]
            regional_holidays = regional_holidays[~regional_holidays.duplicated(keep="first")]
            local_holidays = local_holidays[~local_holidays.duplicated(keep="first")]
            processed_data["is_holiday"] = np.zeros(shape=(len(processed_data), )).astype(int)

            national_indices = processed_data[processed_data["date"].isin(national_holidays["date"])].index
            processed_data["is_holiday"][national_indices] = 1
            for date, state in zip(regional_holidays["date"], regional_holidays["state"]):
                indices = processed_data[((processed_data["date"] == date) & (processed_data["state"] == state))].index
                processed_data["is_holiday"][indices] = 1
            for date, city in zip(local_holidays["date"], local_holidays["city"]):
                indices = processed_data[((processed_data["date"] == date) & (processed_data["city"] == city))].index
                processed_data["is_holiday"][indices] = 1

            processed_data.to_csv(self.datatransformationconfig.processed_data)
        
            logging.info(">>> DATA TRANSFORMATION COMPLETE <<<")
        
        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))

    def split_dataset(self, number_of_test_days = 15):
        logging.info("performing data split for cross-validation")
        try:     
            processed_data = pd.read_csv(self.datatransformationconfig.processed_data)

            last_date = processed_data["date"].iloc[- 1]
            last_date = datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days = number_of_test_days)
            last_date = last_date.strftime("%Y-%m-%d")
            split_index = processed_data[processed_data["date"] == last_date].index[-1]
            train_data = processed_data.iloc[:split_index + 1, :]
            test_data = processed_data.iloc[split_index + 1:, :]

            train_data.to_csv(self.datatransformationconfig.train_data)
            test_data.to_csv(self.datatransformationconfig.test_data)
        
            logging.info("data split complete")

        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))



if __name__ == "__main__":
    datatransformation = DataTransformation()
    datatransformation.transform_data()
    datatransformation.split_dataset(number_of_test_days = 15)
