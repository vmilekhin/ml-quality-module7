from typing import Annotated

from pydantic import BaseModel, Field, FiniteFloat


class PredictRequest(BaseModel):
    values: Annotated[list[FiniteFloat], Field(min_length=4, max_length=128)]
    horizon: Annotated[int, Field(ge=1, le=30)] = 1


class PredictResponse(BaseModel):
    prediction: float
    model_version: str
