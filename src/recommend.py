# import pickle
# from src.config import MODEL_PATH, TOP_K

# model = pickle.load(open(MODEL_PATH, "rb"))

# def recommend(user_id, items):
#     preds = []

#     for item in items:
#         pred = model.predict(user_id, item)
#         preds.append((item, pred.est))

#     preds.sort(key=lambda x: x[1], reverse=True)

#     return preds[:TOP_K]

import pickle
import pandas as pd
from src.config import MODEL_PATH, TOP_K

# Load model
model = pickle.load(open(MODEL_PATH, "rb"))

# Load movie metadata once
movies_df = pd.read_csv("data/movies.csv")

# Create mapping: movieId → title
movie_map = dict(zip(movies_df.movieId, movies_df.title))


def recommend(user_id, items):
    preds = []

    for item in items:
        pred = model.predict(user_id, item)
        preds.append((item, pred.est))

    # Sort by predicted rating
    preds.sort(key=lambda x: x[1], reverse=True)

    top_k = preds[:TOP_K]

    # Convert movieId → movie title
    results = []

    for movie_id, score in top_k:
        movie_name = movie_map.get(movie_id, "Unknown Movie")

        results.append({
            "movie_id": movie_id,
            "movie": movie_name,
            "score": round(score, 3)
        })

    return results