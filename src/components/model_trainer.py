from src.utils.exception import CustomException
from src.utils.logger import logging
from dataclasses import dataclass
from darts.models.forecasting.lgbm import LightGBMModel
import joblib
import os

@dataclass
class ModelTrainerConfig:
    timeseries_data:str = os.path.join("artifacts", "timeseries_data.joblib")
    covariates:str = os.path.join("artifacts", "covariates.joblib")
    oil_model:str = os.path.join("artifacts", "oil_model.joblib")
    trained_model:str = os.path.join("artifacts", "trained_model.joblib")

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig = ModelTrainerConfig()
        logging.info(">>> MODEL TRAINER STARTED <<<")

    def train_model(self):

        """
        This function is responsible for training a model on oil prices for oil forecasts and also for training the final model to forecast sales
        """

        logging.info("executing train_model function")
        try:
            timeseries_data = joblib.load(self.modeltrainerconfig.timeseries_data)
            covariates = joblib.load(self.modeltrainerconfig.covariates)

            logging.info("training LightGBM Model for producing oil forecasts")
            series = covariates[str((1, "AUTOMOTIVE"))]["dcoilwtico"]
            oil_model = LightGBMModel(
                lags = 25,
                output_chunk_length = 1,
                n_estimators = 500
            )
            oil_model.fit(series = series)

            logging.info("creating the final LightGBM Model and training it to forecast sales")

            model = LightGBMModel(
                lags = [-1, -2, -6, -7, -8, -13, -14, -15, -20, -21, -27, -28, -35, -42, -49, -56, -63],
                lags_past_covariates = [-1, -2, -6, -7, -8, -13, -14, -15, -20, -21, -27, -28, -35],
                output_chunk_length = 1,
                n_estimators = 1000
            )
            model.fit(
                series = [timeseries_data[component] for component in timeseries_data.components],
                past_covariates = [covariates[cov] for cov in timeseries_data.components]
            )

            logging.info("saving the trained models to artifacts")

            joblib.dump(oil_model, self.modeltrainerconfig.oil_model)
            joblib.dump(model, self.modeltrainerconfig.trained_model)

            logging.info("models saved successfully")
            logging.info(">>> MODEL TRAINER COMPLETED <<<")

        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))
