import pickle
from pathlib import Path 


# Set project paths
PROJECT_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIRECTORY / "models" / "student_depression_model.pkl"


# Load saved model and vectorizer
with open(MODEL_PATH, "rb") as model_file:
    model_artifacts = pickle.load(model_file)

xgboost_model = model_artifacts["model"]
dict_vectorizer = model_artifacts["vectorizer"]

print("Model and vectorizer loaded successfully.")

def predict_student(student_data):
    # Convert student data into the format expected by the model
    student_record = [student_data]

    # Apply the saved vectorizer
    student_features = dict_vectorizer.transform(student_record)

    # Generate prediction and probability
    predicted_class = xgboost_model.predict(student_features)[0]
    predicted_probability = xgboost_model.predict_proba(student_features)[0, 1]

    # Convert numeric prediction back to its original label
    depression_label = "Yes" if predicted_class == 1 else "No"

    return {
        "depression_prediction": depression_label,
        "depression_probability": round(float(predicted_probability), 4)
    }

if __name__ == "__main__":
    sample_student = {
        "gender": "Female",
        "age": 22,
        "family_background": "Middle class",
        "relationship_status": "Single",
        "marital_status": "Unmarried",
        "living_style": "Bachelor",
        "current_semester": 5,
        "cgpa": 3.5,
        "friend_circle": "Yes",
        "health_condition": "Normal",
        "cost_without_study": 8000,
        "home_division": "Dhaka"
    }

    prediction = predict_student(sample_student)

    print("\nSample student prediction:")
    print(prediction)