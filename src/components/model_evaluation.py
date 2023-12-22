from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils import generate_covariates
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import json
import numpy as np
import pandas as pd
import joblib

@dataclass
class ModelEvaluationConfig:
    trained_model:str = os.path.join("artifacts", "trained_model.joblib")
    oil_model:str = os.path.join("artifacts", "oil_model.joblib")
    covariates:str = os.path.join("artifacts", "covariates.joblib")
    testseries_data:str = os.path.join("artifacts", "testseries_data.joblib")
    timeseries_data:str = os.path.join("artifacts", "timeseries_data.joblib")
    test_covariates:str = os.path.join("artifacts", "test_covariates.joblib")
    results_json:str = os.path.join("artifacts", "results.json")

class ModelEvaluation:
  def __init__(self):
    logging.info(">>> MODEL EVALUATION STARTED <<< ")
    self.modelevaluationconfig = ModelEvaluationConfig()

  def generate_predictions(self):

    """
    This function is responsible for evaluation of the model by forecasting for the next few days
    """

    logging.info("executing the generate_predictions function")
    try:
      trained_model = joblib.load(self.modelevaluationconfig.trained_model)
      oil_model = joblib.load(self.modelevaluationconfig.oil_model)
      covariates = joblib.load(self.modelevaluationconfig.covariates)
      testseries_data = joblib.load(self.modelevaluationconfig.testseries_data)
      test_covariates = joblib.load(self.modelevaluationconfig.test_covariates)
      timeseries_data = joblib.load(self.modelevaluationconfig.timeseries_data)

      logging.info("generating covariates for the next few days after the end of train data")

      oil_forecasts = oil_model.predict(n = len(testseries_data)).pd_series().to_list()
      new_covariates = {}
      for component in testseries_data.components:
        new_covariates[component] = covariates[component].append(
            generate_covariates(
                horizon = len(testseries_data),
                onpromotion = test_covariates[component].pd_dataframe()["onpromotion"],
                oil_forecasts = oil_forecasts,
                is_holiday = test_covariates[component].pd_dataframe()["is_holiday"],
                trained_last_date = oil_model.training_series.end_time()
                )
        )

      logging.info("forecasting for the days equal to number of test data records")

      predictions = trained_model.predict(
          n = len(testseries_data),
          series = [timeseries_data[component] for component in timeseries_data.components],
          past_covariates = [new_covariates[str(component)] for component in timeseries_data.components]
      )

      predictions_df = pd.DataFrame()
      for prediction in predictions:
          predictions_df[prediction.components[0]] = prediction.pd_series().to_list()

      logging.info("returning the train data, test data and predictions for evaluation")

      return timeseries_data.pd_dataframe(), testseries_data.pd_dataframe(), predictions_df
    except Exception as e:
      logging.info(CustomException(e))
      print(CustomException(e))

  def evaluate_predictions(self, train_data, targets, predictions):
    
    """
    This function is responsible for normalisation of the test data and predictions based on the train data
    """

    try:
      logging.info("initialising MinMaxScaler and fitting it on the train data to transform the test set and predictions")

      scaler = MinMaxScaler()
      scaler.fit(np.array(train_data))
      real_values = scaler.transform(np.array(targets))
      predicted_values = scaler.transform(np.array(predictions))

      real = []
      pred = []
      for col in range(real_values.shape[1]):
          real += list(real_values[:, col])
          pred += list(predicted_values[:, col])
      logging.info("writing the model performance report to a JSON file")

      results = {
          "Mean Squared Error (on normalised data)" : mean_squared_error(real, pred),
          "Mean Absolute Error (on normalised data)" : mean_absolute_error(real, pred)
      }

      results_json = json.dumps(results, indent = 3)
      with open(self.modelevaluationconfig.results_json, "w") as jsonfile:
        jsonfile.write(results_json)

      logging.info("model evaluation report saved successfully to artifacts")

    except Exception as e:
      logging.info(CustomException(e))
      print(CustomException(e))