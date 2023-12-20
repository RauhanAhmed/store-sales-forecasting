from src.utils.exception import CustomException
from src.utils.logger import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from hampel import hampel
from darts import TimeSeries
from sklearn.feature_selection import VarianceThreshold
import joblib
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings(action = "ignore")


@dataclass
class DataTransformationConfig:
    oil:str = os.path.join("artifacts", "oil.csv")
    data:str = os.path.join("artifacts", "raw_data.csv")
    stores:str = os.path.join("artifacts", "stores.csv")
    holidays:str = os.path.join("artifacts", "holidays.csv")
    processed_data:str = os.path.join("artifacts", "processed_data.csv")
    train_data:str = os.path.join("artifacts", "train_data.csv")
    test_data:str = os.path.join("artifacts", "test_data.joblib")
    test_data_covariates:str = os.path.join("artifacts", "test_data_covariates.joblib")
    timeseries_data:str = os.path.join("artifacts", "timeseries_data.joblib")
    covariates:str = os.path.join("artifacts", "covariates.joblib")


class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()
        logging.info(">>> DATA TRANSFORMATION STARTED <<<")

    def integrate_data(self):
        
        """
        Function responsible for integrating the datasets without any inconsistencies
        """
        
        logging.info("executing integrate_data function")
        try:
            logging.info("handling missing values")

            oil = pd.read_csv(self.datatransformationconfig.oil)
            data = pd.read_csv(self.datatransformationconfig.data)
            stores = pd.read_csv(self.datatransformationconfig.stores)
            holidays = pd.read_csv(self.datatransformationconfig.holidays)

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

            processed_data.to_csv(self.datatransformationconfig.processed_data, index = False)
        
            logging.info("data integration complete")
        
        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))

    def split_data(self, number_of_test_days = 15):
        
        """
        Function responsible for splitting the data into train and test sets
        """

        logging.info("executing split_data function")
        try:     
            logging.info("performing data split for cross-validation")

            processed_data = pd.read_csv(self.datatransformationconfig.processed_data)

            last_date = processed_data["date"].iloc[- 1]
            last_date = datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days = number_of_test_days)
            last_date = last_date.strftime("%Y-%m-%d")
            split_index = processed_data[processed_data["date"] == last_date].index[-1]
            train_data = processed_data.iloc[:split_index + 1, :]
            test_data = processed_data.iloc[split_index + 1:, :]

            train_data.to_csv(self.datatransformationconfig.train_data, index = False)
            joblib.dump(test_data, self.datatransformationconfig.test_data)
        
            logging.info("data split complete")

        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))

    def transform_data(self):
        
        """
        Function responsible for creating Darts TimeSeries objects for different data series and their respective covariates.
        Also removes outliers using Median Absolute Deviations and removes features with zero variance.
        """

        logging.info("executing transform_data function")
        try:
            train_data = pd.read_csv(self.datatransformationconfig.train_data)
            test_data = joblib.load(self.datatransformationconfig.test_data)

            train_data.drop(["id", "city", "store_type", "state", "cluster"], axis = 1, inplace = True)
            test_data.drop(["id", "city", "store_type", "state", "cluster"], axis = 1, inplace = True)
            
            logging.info("separating different timeseries and their covariates")

            sales = {}
            covariates = {}
            for group, data_slice in train_data.groupby(by = ["store_nbr", "family"]):
                data_slice.set_index("date", drop = True, inplace = True)
                sales_series = data_slice["sales"]
                covariate = data_slice[["onpromotion", "dcoilwtico", "is_holiday"]]
                sales[group] = sales_series
                covariates[str(group)] = covariate

            logging.info("filling missing dates in different series and their covariates")

            series_dataset = pd.DataFrame(data = sales)
            all_dates = set(pd.date_range(start = series_dataset.index[0], end = series_dataset.index[-1]).strftime("%Y-%m-%d"))
            all_missing_dates = all_dates.difference(set(series_dataset.index))
            missing_data = pd.DataFrame(data = {column:[np.NaN] * len(all_missing_dates) for column in series_dataset.columns},\
                                        index = list(all_missing_dates), columns = series_dataset.columns)
            series_dataset = pd.concat([series_dataset, missing_data], axis = 0).sort_index().interpolate()
            for cov in covariates:
                for date in all_missing_dates:
                    covariates[cov].loc[date, :] = [np.NaN] * covariates[cov].shape[1]
                covariates[cov] = covariates[cov].ffill()  

            logging.info("reformatting test_data")

            test_sales = {}
            test_covariates = {}
            for group, data_slice in test_data.groupby(by = ["store_nbr", "family"]):
                data_slice.set_index("date", drop = True, inplace = True)
                test_covariate = data_slice[["onpromotion", "dcoilwtico", "is_holiday"]]
                test_sales_series = data_slice["sales"]
                test_sales[group] = test_sales_series
                test_covariates[str(group)] = test_covariate          

            test_data = pd.DataFrame(data = test_sales)

            logging.info("detecting and removing outliers from different series")    

            temp = series_dataset.apply(lambda x : hampel(x, window_size = 7, n_sigma = 3.0).filtered_data)
            series_dataset = temp.set_index(series_dataset.index)                   

            logging.info("dropping features with zero variances")

            var_threshold = VarianceThreshold(threshold = 0)
            var_threshold.fit(series_dataset)   
            const_features_report = var_threshold.get_support()
            constant_features = []
            for feature, result in zip(series_dataset.columns, const_features_report):
                if result == True:
                    pass
                else:
                    constant_features.append(feature)
            features_to_keep = set(series_dataset.columns).difference(set(constant_features))
            series_dataset = series_dataset[features_to_keep]

            series_dataset = series_dataset[sorted(series_dataset.columns)]
            test_data = test_data[series_dataset.columns]

            logging.info("converting sales series and covariates into Darts TimeSeries")

            series_dataset.set_index(pd.to_datetime(series_dataset.index), inplace = True)
            test_data.set_index(pd.to_datetime(test_data.index), inplace = True)

            timeseries_data = TimeSeries.from_dataframe(series_dataset)
            test_data = TimeSeries.from_dataframe(test_data)

            for cov_key in covariates:
                temp_cov = covariates[cov_key]
                temp_cov.set_index(pd.to_datetime(temp_cov.index), inplace = True)
                covariates[cov_key] = TimeSeries.from_dataframe(temp_cov)

            for cov_key in test_covariates:
                temp_cov = test_covariates[cov_key]
                temp_cov.set_index(pd.to_datetime(temp_cov.index), inplace = True)
                test_covariates[cov_key] = TimeSeries.from_dataframe(temp_cov)      

            joblib.dump(timeseries_data, self.datatransformationconfig.timeseries_data)
            joblib.dump(covariates, self.datatransformationconfig.covariates)
            joblib.dump(test_data, self.datatransformationconfig.test_data)
            joblib.dump(test_covariates, self.datatransformationconfig.test_data_covariates)

            logging.info("saved timeseries_data, test_data and covariates to artifacts")
            logging.info(">>> DATA TRANSFORMATION COMPLETE <<<")

        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))