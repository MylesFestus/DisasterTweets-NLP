from fastapi import FastAPI
import joblib


pipeline = joblib.load("models/best_disaster_tweet_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Disaster tweet API is running"}


@app.post("/predict")
def predict(text: dict):
    review = text["review"]

    prediction = pipeline.predict([review])[0]
    probability = pipeline.predict_proba([review])[0]

    return {"prediction": prediction,
            "probability": float(probability)}
