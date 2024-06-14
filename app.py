# app.py
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()


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
    return {"message": "Welcome to the FastAPI inference service!"}


@app.post("/predict")
async def predict(input: ModelInput):
    # Mock model inference based on age (for simplicity)
    if input.age > 30:
        prediction = ">50K"
    else:
        prediction = "<=50K"
    return {"prediction": prediction}
