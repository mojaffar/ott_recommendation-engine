from surprise import Dataset, Reader, accuracy
from collections import defaultdict

def get_predictions(model, df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    testset = data.build_full_trainset().build_testset()
    return model.test(testset)

def precision_at_k(predictions, k=5, threshold=3.5):
    user_est_true = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []

    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]

        relevant = sum((true >= threshold) for (_, true) in top_k)
        recommended = sum((est >= threshold) for (est, _) in top_k)

        if recommended == 0:
            precisions.append(0)
        else:
            precisions.append(relevant / recommended)

    return sum(precisions) / len(precisions)

def evaluate_model(model, test_df):
    predictions = get_predictions(model, test_df)

    rmse = accuracy.rmse(predictions)
    precision = precision_at_k(predictions)

    return {
        "rmse": rmse,
        "precision@5": precision
    }