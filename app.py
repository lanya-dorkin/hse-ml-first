import io
from typing import List
import joblib
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
import uvicorn
import pandas as pd
from fastapi.responses import StreamingResponse


app = FastAPI()
model = joblib.load('pipeline.pkl')


class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float
    torque: float
    max_torque_rpm: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    brand: str
    seats: int


class Items(BaseModel):
    objects: List[Item]
    

class PredictionResponse(BaseModel):
    selling_price: float


# из задания ручка с одним объектом
# метод post, который получает на вход один объект описанного класса
@app.post("/predict_item")
def predict_item(item: Item) -> PredictionResponse:
    # на вход приходит объект класса Item, поэтому дополнительная валидация не требуется
    df = pd.DataFrame([item.model_dump()])
    return PredictionResponse(selling_price=model.predict(df)[0])


# из задания ручка с несколькими объектами
# метод post, который получает на вход коллекцию объектов описанного класса
@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> StreamingResponse:
    # валидация через Item, приятнее не придумал
    df = pd.read_csv(file.file)
    [Item.model_validate(row.to_dict()) for _, row in df.iterrows()]
    df['selling_price'] = model.predict(df)

    return StreamingResponse(
        io.StringIO(df.to_csv(index=False)),
        media_type='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=predicted_prices.csv'
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)