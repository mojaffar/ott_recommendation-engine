from fastapi import FastAPI
from src.recommend import recommend

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Recommendation API running"}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    items = list(range(1, 100))
    recs = recommend(user_id, items)

    return {
        "user_id": user_id,
        "recommendations": recs
    }