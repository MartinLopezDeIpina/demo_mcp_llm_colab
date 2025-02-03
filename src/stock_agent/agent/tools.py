from stock_agent.agent.models import MaxPriceArgs, AnswerArgs, PriceDateArgs
from langchain_core.tools import StructuredTool


class ToolManager:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def get_max_price_dates_func(self, stock: str, start_date: str, end_date: str) -> float:
        tool_args = locals()
        del tool_args["self"]
        return await self.mcp_client.call_tool("get_max_price_dates_tool", tool_args)

    async def get_min_price_dates_func(self, stock: str, start_date: str, end_date: str) -> float:
        tool_args = locals()
        del tool_args["self"]
        return await self.mcp_client.call_tool("get_min_price_dates_tool", tool_args)

    async def get_price_date_func(self, stock: str, date: str) -> float:
        tool_args = locals()
        del tool_args["self"]
        return await self.mcp_client.call_tool("get_price_date_tool", tool_args)

    async def answer_tool_func(self, value: float) -> float:
        return value

    def get_tool_instances(self):
        max_price_tool = StructuredTool(
            name="get_max_price_dates_tool",
            func=self.get_max_price_dates_func,
            description="Returns the maximum price of a stock between two dates",
            args_schema=MaxPriceArgs
        )
        min_price_tool = StructuredTool(
            name="get_min_price_dates_tool",
            func=self.get_min_price_dates_func,
            description="Returns the minimum price of a stock between two dates",
            args_schema=MaxPriceArgs
        )
        price_tool = StructuredTool(
            name="get_price_date_tool",
            func=self.get_price_date_func,
            description="Returns the price of a stock on a given date",
            args_schema=PriceDateArgs
        )
        answer_tool = StructuredTool(
            name="answer",
            func=self.answer_tool_func,
            description="Answer the question",
            args_schema=AnswerArgs
        )

        return [max_price_tool, min_price_tool, price_tool, answer_tool]

