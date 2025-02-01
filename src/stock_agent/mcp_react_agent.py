from typing import Optional, Any, Sequence, List, Literal
import requests
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, ToolCall, BaseMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.tools import tool, Tool, StructuredTool
import json

from stock_agent.validation_utils import format_messages_into_json, validate_response, get_tool_by_name

initial_prompt_template_str="""Solve the question answering task with interleaving Thought, Action, Observation steps. The action represents the called tool, and the boservation represents the tool's output. You task is to identify the next action based on the observation and the thinking process.

You have 4 possible actions, always think about the best option, only choose one option, answer with the ANSWER tool when you are sure of the result: 
{tools_info}

Your response must contain the action to perform after the thinking process. The action must be in the following json format, inside de \\boxed{{}}, even if the action is the final answer. Remember to use exactly de specified tool schema: 
    \\boxed{{"action": "name_of_the_action_to_perform", "args": {{//required args}}}}

For example, for the action "ANSWER" with the argument "value" with value "000.000", the json would be:
    \\boxed{{"action": "ANSWER", "args": {{"value": "000.000"}}}}

The question is: {question}
Begin!"""

class ToolCallAction(BaseMessage):
    type: str = "tool_call"
    content: str
    tool_name: str
    tool_args: dict

    @classmethod
    def __create__(cls, content: str, tool_name: str, tool_args: dict) -> "ToolCallAction":
        return cls(
            content=content,
            tool_name=tool_name,
            tool_args=tool_args
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
        tools_info = "\n".join([
                                   f"{i + 1}. {tool.name.upper()}: {tool.description}; Required args schema: {tool.args_schema.get_simple_schema()}"
                                   for i, tool in enumerate(self.tools)])

        initial_message_template = HumanMessagePromptTemplate.from_template(initial_prompt_template_str)
        initial_message = initial_message_template.format(
            tools_info=tools_info,
            question=self.question
        )

        return initial_message

    def __init__(
            self,
            llm: BaseLLM,
            tools: List[StructuredTool],
    ):
        self.llm = llm
        self.tools = tools
        self.finished = False
        self.validation_errors = 0
        self.max_validation_errors = 3
        self.messages = []

    def is_finished(self):
        """todo: check if thea agent's scratchpad is too long"""
        return self.finished

    async def run(self, question: str, reset=True):
        self.question = question

        if reset:
            self.messages = []
            self.messages.append(self.create_initial_message())
            self.validation_errors = 0

            self.finished = False

        while not self.is_finished():
            await self.step()

        return self.messages

    def forward(self)-> List[BaseMessage]:
        # Si da muchos problemas gestionar mejor validación
        action = None
        while not action:
            print("Forwarding message")
            response = self.llm.invoke(self.messages)
            print(f"Response: {response}")
            try:
                action = validate_response(response, self.tools)
            except Exception as e:
                print(f"Error validating response: {e}")
                self.validation_errors += 1
                if self.validation_errors > self.max_validation_errors:
                    self.finished = True
                    raise Exception(f"Too many validation errors: {self.validation_errors}, stopping agent")

        new_messages = []

        ai_message = AIMessage(content=f"\n{response['content']}\n")
        new_messages.append(ai_message)
        if action.action.lower() == "answer":
            self.finished = True
            message = AIMessage(f"{action.args['value']}")
            new_messages.append(message)
        else:
            tool = get_tool_by_name(self.tools, action.action)
            tool_args = action.args
            message = ToolCallAction(
                content=f"\n<Action> {tool.name} with args {tool_args} </Action>\n",
                tool_name = tool.name,
                tool_args = tool_args
            )
            new_messages.append(message)

        return new_messages

    async def step(self):
        new_messages = self.forward()
        print(f"New messages: {new_messages}")
        self.messages.extend(new_messages)

        if len(self.messages) > 7:
            self.finished = True

        if type(self.messages[-1]) == ToolCallAction:
            tool_response = await self.execute_tool(self.messages[-1])
            tool_content = f"\n<Observation> {tool_response} </Observation>\n"
            self.messages.append(ToolMessage(content=tool_content, tool_call_id=len(self.messages)))

    async def execute_tool(self, tool_call: ToolCallAction) -> Any:
        try:
            tool = None
            for t in self.tools:
                if t.name == tool_call.tool_name:
                    tool = t
                    break

            result = await tool.invoke(input=tool_call.tool_args)
            value = result.content[0].text

            return value

        except Exception as e:
            return f"Error ejecutando la herramienta: {str(e)}"

