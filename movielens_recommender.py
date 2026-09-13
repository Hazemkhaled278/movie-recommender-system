import pandas as pd

folder_path = r"C:\Users\DELL\downloads\ML\ml-100k"

file_path = f"{folder_path}\\u.data"

column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(file_path, sep="\t", names=column_names)
print(df.head())  







import pandas as pd

folder_path = r"C:\Users\DELL\downloads\ML\ml-100k"
# هنا بنقرا ملف الأفلام وبناخد رقم الفيلم واسمه بس ونسيب الباقي
movies_path = f"{folder_path}\\u.item"
movies_df = pd.read_csv(
    movies_path,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],  
    names=["item_id", "movie_title"],
)
print(movies_df.head())

# هنا بنربط جدول التقييمات بجدول الأفلام عشان يطلع اسم الفيلم جنب التقييم بدل رقمه
merged_df = pd.merge(df, movies_df, on="item_id")
print(merged_df.head())




# نظبط مكتبة Surprise ونقسم الداتا لحتت تدريب واختبار، ونبدأ ندرب نموذج الـ SVD
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
reader = Reader(rating_scale=(1, 5))
folder_path = r"C:\Users\DELL\downloads\ML\ml-100k"
file_path = f"{folder_path}\\u.data"
data = Dataset.load_from_file(file_path, reader=reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
print("RMSE (Prediction error ):", accuracy.rmse(predictions))

# بنجرب نتوقع تقييم مستخدم معين (رقم 196) لفيلم معين (رقم 302) هيدي كام
user_id_str = "196" 
item_id_str = "302"

prediction = model.predict(user_id_str, item_id_str)
print(f"User {user_id_str} predicted rating for item {item_id_str}:")
print(f"Estimated Rating (Est): {prediction.est:.2f}")

# بنقيس نسبة الخطأ (RMSE) تاني للاطمئنان
predictions = model.test(testset)
print("RMSE (Root Mean Squared Error):", accuracy.rmse(predictions))




#هنا هنطلع أحسن 10 أفلام للمستخدم وتستبعد اللي شافه قبل كده
import pandas as pd


def get_top_n_recommendations(model, user_id, movies_df, ratings_df, n=10):
  all_item_ids = movies_df["item_id"].unique()

  rated_items = ratings_df[ratings_df["user_id"] == int(user_id)][
      "item_id"
  ].values
  unrated_items = [
      item for item in all_item_ids if item not in rated_items
  ]

  predictions = []
  for item_id in unrated_items:
    pred = model.predict(str(user_id), str(item_id))
    predictions.append((item_id, pred.est))

  predictions.sort(key=lambda x: x[1], reverse=True)

  top_n = predictions[:n]

  top_movie_ids = [item[0] for item in top_n]
  top_movie_ratings = [item[1] for item in top_n]

  recommendations_df = movies_df[
      movies_df["item_id"].isin(top_movie_ids)
  ].copy()
  recommendations_df["predicted_rating"] = top_movie_ratings
  recommendations_df = recommendations_df.sort_values(
      by="predicted_rating", ascending=False
  )

  return recommendations_df[[ "movie_title", "predicted_rating"]]

# هنفذ function ال على المستخدم رقم 196 ونطبع له الترشيحات
target_user_id = 196
top_10 = get_top_n_recommendations(
    model, target_user_id, movies_df, df, n=10
)
print(f"Top 10 Movie Recommendations for User {target_user_id}:")
print(top_10.to_string(index=False))

#هنحسب نسبة الدقة والاسترجاع

from collections import defaultdict
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """Return precision and recall at k metrics for each user."""

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = (
            n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        )

        recalls[uid] = (
            n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        )

    return precisions, recalls

#هنحسب ال avg النهائي و نطبعها
precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4.0)
mean_precision = sum(precisions.values()) / len(precisions)
mean_recall = sum(recalls.values()) / len(recalls)

print(f"Mean Precision@10: {mean_precision:.4f}")
print(f"Mean Recall@10:    {mean_recall:.4f}")



#هنا بنحفظ كل اللي مودل عاملة عشان نقدر نستخدمة وقت ما نحب و يكون جاهز
import pickle

with open("svd_movie_model.pkl", "wb") as f:
  pickle.dump(model, f)

with open("movies_df.pkl", "wb") as f:
  pickle.dump(movies_df, f)

with open("ratings_df.pkl", "wb") as f:
  pickle.dump(df, f)

print("Model and data saved successfully using pickle!")