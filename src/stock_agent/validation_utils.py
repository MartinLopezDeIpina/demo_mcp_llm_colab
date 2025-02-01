from typing import  List, Literal
from langchain_core.tools import tool, Tool
from pydantic import ValidationError
import json
import re
from stock_agent.models import Action

def extract_boxed_content(response:dict) -> dict or None:
    response = response["content"]
    pattern = r'\\boxed\{(.*?\}+)}'
    try:
        match = re.search(pattern, response, re.DOTALL)
    except Exception as e:
        print(f"Error in re.search: {e}")
        return None

    if match:
        return match.group(1)
    print("No match for boxed content")
    return None

def parse_response(response: dict):
    boxed_response = extract_boxed_content(response)

    if boxed_response:
        boxed_response = f"{{{boxed_response}}}"
        try:
            json_response = json.loads(boxed_response)
        except Exception as e:
            print(f"Error parsing json: {e}")
            return None
        return json_response

    return None

def get_tool_by_name(available_tools: List[Tool], tool_name: str):
    for tool in available_tools:
        if tool.name == tool_name.lower():
            return tool
    return None

def validate_response(response: dict, available_tools: List[Tool]):
    parsed_response = parse_response(response)

    if parsed_response:
        try:
            validated_action = Action.model_validate(parsed_response)
            tool = get_tool_by_name(available_tools, validated_action.action)
            if tool:
                tool.args_schema.model_validate(validated_action.args)
            return validated_action

        except ValidationError as e:
            print(f"Pydantic validation error: {e}")
            return None
    else:
        return None

def format_messages_into_json(messages):
    json_messages = []
    for message in messages:
        # los tool calls se guardan como diccionarios
        if type(message) == dict:
            json_messages.append(message)
        else:
            json_messages.append(message.model_dump())
    return json_messages
