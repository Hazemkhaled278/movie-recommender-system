# Movie Recommendation Engine (Collaborative Filtering)

A professional machine learning project that builds a personalized movie recommendation system using the **MovieLens 100k** dataset and Collaborative Filtering techniques.

---

# Project Overview

This project processes user ratings and movie metadata to train a Matrix Factorization model (**SVD**). It predicts user ratings for unseen movies and generates top-N personalized movie recommendations while filtering out movies the user has already rated.

---

# Tech Stack & Libraries

- **Python** (Core programming language)
- **Pandas** (Data manipulation and cleaning)
- **scikit-surprise** (Collaborative filtering and recommendation algorithms)
- **Pickle** (Model persistence and serialization)

---

# Model Performance & Metrics

The model was evaluated using an 80/20 train-test split, yielding the following results:

- **RMSE (Root Mean Squared Error):** ~0.9367
- **Mean Precision@10:** 0.8764
- **Mean Recall@10:** 0.3026

---

# Project Structure

````text
├── ml-100k/                 # Dataset folder (u.data, u.item)
├── movielens_recommender.py # Main python script (training & recommendation logic)
├── svd_movie_model.pkl      # Saved trained SVD model
├── movies_df.pkl            # Processed movies dataframe
├── ratings_df.pkl           # Processed ratings dataframe
└── README.md                # Project documentation

---

# How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas scikit-learn scikit-surprise

   python movielens_recommender.py

   👨‍💻 Developed by Hazem Mohamed
````
