from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils import generate_covariates
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import joblib

@dataclass
class ModelEvaluationConfig:
    trained_model_path:str = os.path.join("artifacts", "trained_model.joblib")
    oil_model_path:str = os.path.join("artifacts", "oil_model.joblib")
    covariates:str = os.path.join("artifacts", "covariates.joblib")
    test_data_path:str = os.path.join("artifacts", "test_data.joblib")
    timeseries_data_path:str = os.path.join("artifacts", "timeseries_data.joblib")
    test_covariates_path:str = os.path.join("artifacts", "test_data_covariates.joblib")

class ModelEvaluation:
    def __init__(self):
        self.modelevaluationconfig = ModelEvaluationConfig()

    def generate_predictions(self):
        try:
            trained_model = joblib.load(self.modelevaluationconfig.trained_model_path)
            oil_model = joblib.load(self.modelevaluationconfig.oil_model_path)
            covariates = joblib.load(self.modelevaluationconfig.covariates)
            test_data = joblib.load(self.modelevaluationconfig.test_data_path)
            test_data_covariates = joblib.load(self.modelevaluationconfig.test_covariates_path)
            timeseries_data = joblib.load(self.modelevaluationconfig.timeseries_data_path)

            oil_forecasts = oil_model.predict(n = len(test_data)).pd_series().to_list()
            new_covariates = [
                covariates[cov].append(generate_covariates(
                    horizon = len(test_data),
                    onpromotion = test_data_covariates[cov].pd_dataframe()["onpromotion"],
                    oil_forecasts = oil_forecasts,
                    is_holiday = test_data_covariates[cov].pd_dataframe()["is_holiday"],
                    trained_last_date = oil_model.training_series.end_time()
                )) for cov in test_data.components
            ]

            predictions = trained_model.predict(
                n = len(test_data),
                series = [timeseries_data[series] for series in timeseries_data.components],
                past_covariates = new_covariates
            )

            predictions_df = pd.DataFrame()
            for prediction in predictions:
                predictions_df[prediction.components[0]] = list(prediction.pd_series())

            return timeseries_data, test_data, predictions_df
        except Exception as e:
            print(CustomException(e))

    def evaluate_predictions(self, train_data, targets, predictions):
        try:
            scaler = MinMaxScaler()
            scaler.fit(np.array(train_data))
            real_values = scaler.transform(np.array(targets))
            predicted_values = scaler.transform(np.array(predictions))

            real = []
            pred = []
            for col in range(real_values.shape[1]):
                real += real_values[:, col]
                pred += predicted_values[:, col]

            logging.info(mean_squared_error(real, pred))
        except Exception as e:
            print(CustomException(e))
