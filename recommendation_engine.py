import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Collaborative Filtering Setup
ratings = pd.read_csv("user_product_ratings.csv")  # columns: userID, productID, rating
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userID', 'productID', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)
svd_model = SVD()
svd_model.fit(trainset)

# Gradient Boosting with metadata
features = pd.read_csv("user_product_features.csv")  # columns: age, clicks, etc., target: purchase
X = features.drop("purchase", axis=1)
y = features["purchase"]

clf = GradientBoostingClassifier()
clf.fit(X, y)

print("Recommendation model training complete.")
