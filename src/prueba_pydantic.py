import json
from pydantic import BaseModel, Field

class Action(BaseModel):
    action: str = Field(title="Action", description="Action to perform")
    args: dict = Field(title="Arguments", description="Arguments for the action")

schema_json = Action.model_json_schema()
print(json.dumps(schema_json, indent=2))
