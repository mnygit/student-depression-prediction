import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier 

# Set project paths
PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIRECTORY / "data" / "Student_Depression_Dataset.csv"
MODEL_DIRECTORY = PROJECT_DIRECTORY / "models"
MODEL_PATH = MODEL_DIRECTORY / "student_depression_model.pkl"


# Load dataset
student_data = pd.read_csv(DATASET_PATH)

# Standardize column names
student_data.columns = student_data.columns.str.lower()

print(f"Dataset shape: {student_data.shape}")
print("\nTarget distribution:")
print(student_data["depression"].value_counts())

# Separate features and target
target = student_data["depression"]

features = student_data.drop(columns=["depression"])

# Encode target labels
target = target.map({"No": 0, "Yes": 1}).values

# Convert features into dictionaries
training_records = features.to_dict(orient="records")

# Fit and apply the DictVectorizer
dict_vectorizer = DictVectorizer(sparse=False)

X_train = dict_vectorizer.fit_transform(training_records)
y_train = target

print(f"\nFeature matrix shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")

# Train the final XGBoost model
xgboost_model = XGBClassifier(
    eta=0.1,
    max_depth=6,
    min_child_weight=1,
    random_state=42,
    eval_metric="logloss"
)

xgboost_model.fit(X_train, y_train)

print("\nXGBoost model training completed.")


MODEL_DIRECTORY.mkdir(exist_ok=True)

# Save model and vectorizer together
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(
        {
            "model": xgboost_model,
            "vectorizer": dict_vectorizer
        },
        model_file
    )

print(f"\nModel and vectorizer saved to: {MODEL_PATH}")