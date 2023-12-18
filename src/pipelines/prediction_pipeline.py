from src.utils import generate_covariates
from src.utils.exception import CustomException
import joblib
import os
from typing import List
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    oil_model_path:str = os.path.join("artifacts", "oil_model.joblib")
    trained_model_path:str = os.path.join("artifacts", "trained_model.joblib")
    covariates_path:str = os.path.join("artifacts", "covariates.joblib")
    timeseries_data_path:str = os.path.join("artifacts", "timeseries_data.joblib")

class PredictionPipeline:
    def __init__(self):
        self.predictionpipelineconfig = PredictionPipelineConfig()

    def produce_forecasts(
        self,
        store_nbr:int,
        family:str,
        horizon:int,
        onpromotion:List[int],
        is_holiday:List[int],
    ):
        try:
            # Loading models and previous covariates
            oil_model = joblib.load(self.predictionpipelineconfig.oil_model_path)
            trained_model = joblib.load(self.predictionpipelineconfig.trained_model_path)
            covariates = joblib.load(self.predictionpipelineconfig.covariates_path)
            timeseries_data = joblib.load(self.predictionpipelineconfig.timeseries_data_path)

            # generating oil forecasts for 30 days
            oil_forecasts = oil_model.predict(n = 30).pd_series().to_list()

            # generating new past covariates for final model
            covariate = generate_covariates(
                horizon = horizon,
                onpromotion = onpromotion,
                oil_forecasts = oil_forecasts,
                is_holiday = is_holiday,
                trained_last_date = oil_model.training_series.end_time()
            )

            series_name = str((store_nbr, family))
            new_covariates = covariates[series_name].append(covariate)

            # generating sales predictions for the supplied forecast horizon
            predictions = trained_model.predict(
                n = horizon,
                series = timeseries_data[series_name],
                past_covariates = new_covariates
            )

            return [round(x, 2) for x in predictions.pd_series().to_list()]
        except Exception as e:
            print(CustomException(e))
