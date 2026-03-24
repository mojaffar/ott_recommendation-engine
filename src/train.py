import pickle
from surprise import Dataset, Reader, SVD
from src.data_processing import load_data, clean_data, train_test_split
from src.evaluate import evaluate_model
from src.config import MODEL_PATH

def train():
    df = load_data("data/ratings.csv")
    df = clean_data(df)

    train_df, test_df = train_test_split(df)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        train_df[['userId', 'movieId', 'rating']], reader
    )

    trainset = data.build_full_trainset()

    model = SVD()
    model.fit(trainset)

    metrics = evaluate_model(model, test_df)
    print("Evaluation:", metrics)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train()