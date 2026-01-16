from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="ML Classification API",
    description="API phân loại hoa Iris sử dụng FastAPI",
    version="1.0"
)

class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Chiều dài đài hoa (cm)")
    sepal_width: float  = Field(..., gt=0, description="Chiều rộng đài hoa (cm)")
    petal_length: float = Field(..., gt=0, description="Chiều dài cánh hoa (cm)")
    petal_width: float  = Field(..., gt=0, description="Chiều rộng cánh hoa (cm)")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

model_path = "model/model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Không tìm thấy file model.pkl. Hãy chạy train_model.py trước.")

model = joblib.load(model_path)
class_names = ["Setosa", "Versicolor", "Virginica"]

@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        features = [[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]]
        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)
        
        class_index = int(prediction[0])
        confidence = float(np.max(prediction_prob))

        return {
            "prediction": class_names[class_index],
            "class_id": class_index,
            "confidence": round(confidence, 4),
            "message": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root():
    return {"status": "API is running"}