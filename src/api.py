from http.client import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained pipeline
pipeline = joblib.load("models/best_disaster_tweet_model.pkl")

app = FastAPI(
    title="Disaster tweet Classification API",
    description="Predict whether a tweet is disaster or not.",
    version="1.0"
)

# Request schema
class TweetRequest(BaseModel):
    text: str

# Response schema
class PredictionRequest(BaseModel):
    disaster: int
    probability: float


@app.get("/")
def home():
    return {"message": "Disaster tweet API is running"}


@app.post("/predict", response_model=PredictionRequest)
def predict(request: TweetRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Please provide a text"
        )


    prediction = int(pipeline.predict([text])[0])
    probability = max(pipeline.predict_proba([text])[0])

    return {"disaster": prediction,
            "probability": float(probability)}
