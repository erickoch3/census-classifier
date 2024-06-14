# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from src import model
from src import train_model
from src import data as datalib

app = FastAPI()


model, encoder, lb = model.load_model(model.PROD_MODEL_FILENAME, model.MODEL_FOLDER)

class ModelInput(BaseModel):
    age: int = Field(...)
    workclass: str = Field(...)
    fnlgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 25,
                "workclass": "Private",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }
    )


@app.get("/")
async def read_root():
    return {"message": "Welcome to Eric's FastAPI inference service! Let's predict someone's income."}


@app.post("/predict")
async def predict(input: ModelInput):
    # Mock model inference based on age (for simplicity)
    if input.age > 30:
        prediction = ">50K"
    else:
        prediction = "<=50K"
    return {"prediction": prediction}

@app.post("/predict")
async def predict(input: ModelInput):
    try:
        # Convert input to the required format for the model
        input_data = input.dict(by_alias=True)
        input_df = datalib.prepare_input_data(input_data)  # Define this function in datalib
        encoded_input, _, _, _ = datalib.process_data(
            input_df,
            categorical_features=train_model.CAT_FEATURES,
            training=False,
            encoder=encoder
        )
        # Perform inference
        preds = model.inference(model, encoded_input)
        prediction = lb.inverse_transform(preds)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
