from src.pipelines.prediction_pipeline import PredictionPipeline
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

app = FastAPI()

class Covariate_params(BaseModel):
    store_nbr: int
    family: str
    horizon: int
    onpromotion: List[int]
    is_holiday: List[int]


@app.post("/")
async def get_forecasts(params: Covariate_params):
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