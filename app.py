from src.pipelines.prediction_pipeline import PredictionPipeline
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

# initialising FastAPI
app = FastAPI()

# creating the class inheriting the BaseModel class for custom data types
class Covariate_params(BaseModel):
    store_nbr: int
    family: str
    horizon: int
    onpromotion: List[int]
    is_holiday: List[int]

# creating API method
@app.post("/")
async def get_forecasts(params: Covariate_params):
    """
    This endpoint fetches the input data provided by the user and returns the generated forecasts by the model.
    It is backed by the LightGBM Models trained on multiple multivariate timeseries data of historical store sales.

    ***Parameters***
    A JSON object containing information in the following fashion :
    - store_nbr: integer -> The store number whose forecasts are to be made
    - family: string -> An uppercase string of department name
    - horizon: integer -> The forecast horizon or the number of days of sales we need to forecast
    - onpromotion: list[integer] -> A list of integers with the number of items on promotion each day
    - is_holiday: list[integer] -> A list of binary integers (0, 1) denoting a holiday or not on a day  

    ***API Response***
    An Ndarray representing the responses for each input JSON object with sales returned in floating point numbers 
    """
    pipeline_obj = PredictionPipeline()

    store_nbr = params.store_nbr
    family = params.family
    horizon = params.horizon
    onpromotion = params.onpromotion
    is_holiday = params.is_holiday

    response = pipeline_obj.produce_forecasts(
        store_nbr = store_nbr,
        family = family,
        horizon = horizon,
        onpromotion = onpromotion,
        is_holiday = is_holiday
    )
    
    return response