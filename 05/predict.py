import pickle

from typing import Dict, Any
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="predictions")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict(data: Dict[str, Any]):
    result = pipeline.predict_proba(data)[0, 1]
    return {
            "churn_probability": float(result),
            "churn": bool(float(result) >= 0.5)
        }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
