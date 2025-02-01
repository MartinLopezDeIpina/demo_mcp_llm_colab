from pydantic import BaseModel, Field
from typing import Literal

class Action(BaseModel):
    action: str = Field(title="Action", description="Action to perform")
    args: dict = Field(title="Arguments", description="Arguments for the action")

class SimpleSchemaArgs(BaseModel):
    @classmethod
    def get_simple_schema(cls) -> dict:
        schema = cls.model_json_schema()["properties"]
        for name, value in schema.items():
            value.pop("title")
        return schema

class MaxPriceArgs(SimpleSchemaArgs):
    stock: Literal["Amazon", "Google", "Nvidia", "Meta", "Sp500", "Apple"] = Field(
        description="Stock name"
    )
    start_date: str = Field(
        description="Start date (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    end_date: str = Field(
        description="End date (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )

class PriceDateArgs(SimpleSchemaArgs):
    stock: Literal["Amazon", "Google", "Nvidia", "Meta", "Sp500", "Apple", "Tesla"] = Field(
        description="Stock name"
    )
    date: str = Field(
        description="Date (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )

class AnswerArgs(SimpleSchemaArgs):
    value: float = Field(
        description="Answer value"
    )


