from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoringItem(BaseModel):
    VasInv: int
    NumberTumor: int
    SingleTumor: float

with open('model_pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    nparr = df.to_numpy()
    yhat = model.predict(nparr)
    answer = yhat[0]
    answer = round(answer, 2)
    return {"prediction": answer}