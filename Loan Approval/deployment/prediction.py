import joblib
import pandas as pd

def load_model(model_path):
    return joblib.load(model_path)

def predict_attrition(model, data):
    predictions = model.predict(data)
    return ["Approved" if pred == 1 else "Rejected" for pred in predictions]
