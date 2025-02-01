import asyncio

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from typing import Literal
from client import MCPClient

mcp_client = MCPClient()


class MaxPriceArgs(BaseModel):
    stock: Literal["Amazon", "Google", "Nvidia", "Meta", "sp500", "apple"] = Field(
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

    @classmethod
    def get_simple_schema(cls) -> dict:
        schema = cls.model_json_schema()["properties"]
        for name, value in schema.items():
            value.pop("title")
        return schema


async def get_max_price_dates_tool(stock: str, start_date: str, end_date: str) -> float:
    tool_args = {
        "stock": stock,
        "start_date": start_date,
        "end_date": end_date
    }
    return await mcp_client.call_tool("get_max_price_dates_tool", tool_args)

max_price_tool = StructuredTool(
    name="get_max_price_dates_tool",
    func=get_max_price_dates_tool,
    description="Returns the maximum price of a stock between two dates",
    args_schema=MaxPriceArgs
)



async def main():
    await mcp_client.connect_to_server()

    tool_call = {
        "tool_name": "get_max_price_dates_tool",
        "args": {
            "stock": "sp500",
            "start_date": "2024-06-16",
            "end_date": "2024-06-20"
        }
    }
    result = await max_price_tool.invoke(input=tool_call["args"])
    value = result.content
    value = value[0].text
    value = float(value)

    print(value)

    await mcp_client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())