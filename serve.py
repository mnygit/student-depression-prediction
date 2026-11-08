from fastapi import FastAPI
from pydantic import BaseModel 

from predict import predict_student


# Create the FastAPI application
app = FastAPI()


# Define the expected student input fields
class StudentData(BaseModel):
    gender: str
    age: int
    family_background: str
    relationship_status: str
    marital_status: str
    living_style: str
    current_semester: int
    cgpa: float
    friend_circle: str
    health_condition: str
    cost_without_study: float
    home_division: str


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict(student: StudentData):
    student_data = student.model_dump()
    return predict_student(student_data)