import asyncio
from typing import Optional, Any, Sequence, List, Literal

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, ToolCall, BaseMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, Tool, StructuredTool
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

def get_tool_by_name(available_tools: List[Tool], tool_name: str):
    for tool in available_tools:
        if tool.name == tool_name.lower():
            return tool
    return None

def validate_response(response: str, available_tools: List[Tool]):
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

initial_prompt_template_str="""You are a stock market analyst. You have been asked a question about the stock market price, answer precisely and accurately.

You have 4 possible actions, always think about the best option, only choose one option, remember to use the tools to help you: 
{tools_info}

Your response must contain the action to perform after the thinking process. The action must be in the following json format, even if the action is the final answer. Remember to use exactly de specified tool schema: 
    \\boxed{{"action": "name_of_the_action_to_perform", "args": {{//required args}}}}

For example, for the action "ACTION" with the argument "arg1" with value "value", the json would be:
    \\boxed{{"action": "ACTION", "args": {{"arg1": "value"}}}}

The question is: {question}"""

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

class AnswerArgs(SimpleSchemaArgs):
    value: float = Field(
        description="Answer value"
    )

    @classmethod
    def get_simple_schema(cls) -> dict:
        schema = cls.model_json_schema()["properties"]
        for name, value in schema.items():
            value.pop("title")
        return schema


async def get_max_price_dates_tool(stock: str, start_date: str, end_date: str) -> float:
    tool_args = locals()
    return await mcp_client.call_tool("get_max_price_dates_tool", tool_args)

async def answer_tool_func(value: float) -> float:
    return value

max_price_tool = StructuredTool(
    name="get_max_price_dates_tool",
    func=get_max_price_dates_tool,
    description="Returns the maximum price of a stock between two dates",
    args_schema=MaxPriceArgs
)

answer_tool = StructuredTool(
    name="answer",
    func=answer_tool_func,
    description="Answer the question",
    args_schema=AnswerArgs
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

        return response


class ReactAgent:
    def create_initial_message(self):
        tools_info = "\n".join([f"{i+1}. {tool.name.upper()}: {tool.description}; Required args schema: {tool.args_schema.get_simple_schema()}" for i, tool in enumerate(self.tools)])

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


    async def run(self, reset=True):
        if reset:
            self.messages = []
            self.messages.append(self.create_initial_message())

            self.finished = False

        while not self.is_finished():
            await self.step()

        return self.messages[-1]

    def forward(self):
        # Si da muchos problemas gestionar mejor validación
        action = None
        while not action:
            print("Forwarding message")
            response = self.llm.invoke(self.messages)
            try:
                action = validate_response(response, self.tools)
            except Exception as e:
                print(f"Error validating response: {e}")

        if action.action.lower() == "answer":
            self.finished = True
            message = AIMessage("{action.args['value']}")
        else:
            tool = get_tool_by_name(self.tools, action.action)
            tool_args = action.args
            message = ({
                "name": tool.name,
                "args": tool_args,
                "id": len(self.messages),
                "type": "tool_call"
            })

        return message

    async def step(self):
        new_message = self.forward()
        print(f"New message: {new_message}")
        self.messages.append(new_message)

        if type(new_message) == dict and new_message["type"] == "tool_call":
            tool_response = await self.execute_tool(new_message)
            self.messages.append(ToolMessage(content=tool_response, tool_call_id="1"))

    async def execute_tool(self, tool_call: ToolCall) -> Any:
        try:
            tool = None
            for t in self.tools:
                if t.name == tool_call["name"]:
                    tool = t
                    break

            result = await tool.invoke(input=tool_call["args"])
            value = result.content[0].text

            return value
            
        except Exception as e:
            return f"Error ejecutando la herramienta: {str(e)}"

async def main():
    await mcp_client.connect_to_server()
    url_grok = "https://89e9-34-16-215-63.ngrok-free.app"
    url_grok = f"{url_grok}/generate_messages"

    colab_llm = CustomColabLLM(colab_url=url_grok)
    react_mcp_agent = ReactAgent(
            llm=colab_llm,
            tools=[max_price_tool, answer_tool],
            question="What was the top price of amazon stock during 2024-12-18 to 2025-04-07?"
            )
    last_message = await react_mcp_agent.run(reset=True)
    print(last_message)

if __name__ == "__main__":
    asyncio.run(main())
