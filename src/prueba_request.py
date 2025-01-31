from typing import Optional, Any, Sequence, List

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, ToolCall, BaseMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, Tool
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
import json
import re

from client import MCPClient

mcp_client = MCPClient()

class Action(BaseModel):
    action: str = Field(title="Action", description="Action to perform")
    args: dict = Field(title="Arguments", description="Arguments for the action")

def extract_boxed_content(response:dict) -> dict:
    response = response["content"]
    pattern = r'\\boxed\{(.*?\}+)}'
    try:
        match = re.search(pattern, response, re.DOTALL)
    except Exception as e:
        print(f"Error in re.search: {e}")
        return None

    if match:
        return match.group(1)

    return None

def parse_response(response: str):
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

def validate_response(response: str):
    parsed_response = parse_response(response)

    if parsed_response:
        try:
            validated_action = Action.parse_obj(parsed_response)
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



initial_prompt_template_str="""You are a stock market analyst. You have been asked a question about the stock market price, answer precisely and accurately.

You have 4 possible actions, always think about the best option, only choose one option, remember to use the tools to help you: 
    {tools_info}
    4. ANSWER: Anwer the question with the precise value.

Your response must contain the action to perform after the thinking process. The action must be in the following json format. Remember to use exactly de specified tool schema: 
    \\boxed{{"action": "name_of_the_action_to_perform", "args": {{//required args}}}}

For example, for the action "ACTION" with the argument "arg1" with value "value", the json would be:
    \\boxed{{"action": "ACTION", "args": {{"arg1": "value"}}}}

The question is: {question}"""

class MaxPriceArgs(BaseModel):
    stock: str = Field(..., description="Stock name")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")

    @classmethod
    def get_simple_schema(cls) -> dict:
        schema = cls.model_json_schema()["properties"]
        for name, value in schema.items():
            value.pop("title")
        return schema


async def get_max_price_dates_tool(stock: str, start_date: str, end_date: str) -> float:
    tool_args = {
        "stock": stock,
        "fecha1": start_date,
        "fecha2": end_date
    }
    return await mcp_client.call_tool("get_max_price_dates", tool_args)

max_price_tool = Tool(
    name="get_max_price_dates",
    func=get_max_price_dates_tool,
    description="Returns the maximum price of a stock between two dates",
    args_schema=MaxPriceArgs
)


class CustomColabLLM:
    def __init__(self, colab_url: str):
        self.colab_url = colab_url

    def invoke(self, messages: List[BaseMessage]):
        json_messages = format_messages_into_json(messages)
        params = {
                "messages": json.dumps(json_messages)
                }
        response = requests.get(self.colab_url, params=params).json()

        try:
            #todo: validar también cada tool el nombre y sus argumentos -> si no set en la lista mal, si está y args mal también malá
            validated_action = validate_response(response)
            return validated_action
        except Exception as e:
            print(f"Error validating response: {str(e)}")
            return None

class ReactAgent:
    def create_initial_message(self):
        tools_info = "\n".join([f"{i+1}. {tool.name.upper()}: {tool.description}; Required args: {tool.args_schema.get_simple_schema()}" for i, tool in enumerate(self.tools)])

        initial_message_template = HumanMessagePromptTemplate.from_template(initial_prompt_template_str)
        initial_message = initial_message_template.format(
            tools_info=tools_info,
            question=self.question
        )

        return initial_message

    def __init__(
            self,
            llm: BaseLLM,
            tools: List[Tool],
            question: str,
            ):

        self.llm = llm
        self.tools = tools
        self.question = question
        self.finished = False

    def is_finished(self):
        """todo: check if thea agent's scratchpad is too long"""
        return self.finished


    def run(self, reset=True):
        if reset:
            self.messages = []
            self.messages.append(self.create_initial_message())

            self.finished = False

        while not self.is_finished():
            self.step()

        return self.messages[-1]

    def forward(self):
        # Si da muchos problemas gestionar mejor validación
        action = None
        while not action:
            action = self.llm.invoke(self.messages)

        if action.action.lower() == "answer":
            self.finished = True
            return AIMessage("{action.args['value']}")
        elif action.action.lower() in [tool.name for tool in self.tools]:
            for tool in self.tools:
                if tool.name == action.action.lower():
                    tool_args = action.args
                    ToolCall = ({
                        "name": tool.name,
                        "args": tool_args,
                        "id": "1",
                        "type": "tool_call"
                    })
                    return ToolCall

    def step(self):
        new_message = self.forward()
        self.messages.append(new_message)

        if new_message["type"] == "tool_call":
            tool_response = self.execute_tool(new_message)
            self.messages.append(ToolMessage(content=tool_response, tool_call_id="1"))

    def execute_tool(self, tool_call: ToolCall) -> Any:
        try:
            # Encontrar la función original (no el objeto Tool)
            tool_func = None
            for tool in self.tools:
                if tool.name == tool_call["name"]:
                    tool_func = tool.func 
                    break
                    
            if tool_func is None:
                raise ValueError(f"Herramienta {tool_call["name"]} no encontrada")

            result = tool_func(**tool_call["args"])
            return result
            
        except Exception as e:
            return f"Error ejecutando la herramienta: {str(e)}"

url_grok = "https://c437-34-16-156-112.ngrok-free.app"
url_grok = f"{url_grok}/generate_messages"

colab_llm = CustomColabLLM(colab_url=url_grok)
react_mcp_agent = ReactAgent(
        llm=colab_llm,
        tools=[max_price_tool],
        question="What was the top price of amazon stock during 2024-12-18 to 2025-04-07?"
        )
last_message = react_mcp_agent.run(reset=True)
print(last_message)

