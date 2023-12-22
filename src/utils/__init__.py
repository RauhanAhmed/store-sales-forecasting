from pandas import Timestamp
from datetime import timedelta
from darts import TimeSeries
import pandas as pd
from typing import List

def generate_covariates(
        horizon:int,
        onpromotion:List[int],
        oil_forecasts:List[float],
        is_holiday:List[int],
        trained_last_date:Timestamp
    ):

    """
    This function is responsible for generating past covariates for forecasting of sales
    """

    if horizon > 30:
        return "Forecast horizon cannot be greater than 30"
    elif horizon <= 0:
        return "Forecast horizon must be positive"
    elif (horizon > len(onpromotion)) | (horizon > len(is_holiday)):
        return "Length mismatch"
    else:
        new_covariates = pd.DataFrame(data = {
            "onpromotion":onpromotion[:horizon],
            "dcoilwtico":oil_forecasts[:horizon],
            "is_holiday":is_holiday[:horizon]
        })

        last_date = trained_last_date
        start_date = last_date + timedelta(days = 1)
        end_date = last_date + timedelta(days = horizon)
        new_indices = pd.date_range(start = start_date, end = end_date)
        new_covariates.set_index(new_indices, inplace = True)
        new_covariates.index.name = "date"
        new_covariates = TimeSeries.from_dataframe(new_covariates)

    return new_covariates
