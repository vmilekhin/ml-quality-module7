from fastapi import FastAPI, HTTPException

from app.predictor import predict
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Forecast service", version="1.0")


@app.post("/predict", response_model=PredictResponse)
def predict_route(request: PredictRequest) -> PredictResponse:
    """Return a deterministic baseline forecast."""
    try:
        result = predict(request.values, request.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(prediction=result, model_version="baseline-1")
