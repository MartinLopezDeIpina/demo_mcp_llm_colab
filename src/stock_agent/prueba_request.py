import asyncio
from langchain_core.tools import tool, Tool, StructuredTool
from client import MCPClient
from stock_agent.mcp_react_agent import CustomColabLLM, ReactAgent
from stock_agent.models import MaxPriceArgs, AnswerArgs, PriceDateArgs

mcp_client = MCPClient()

async def get_max_price_dates_func(stock: str, start_date: str, end_date: str) -> float:
    tool_args = locals()
    return await mcp_client.call_tool("get_max_price_dates_tool", tool_args)

async def get_min_price_dates_func(stock: str, start_date: str, end_date: str) -> float:
    tool_args = locals()
    return await mcp_client.call_tool("get_min_price_dates_tool", tool_args)

async def get_price_date_func(stock: str, date: str) -> float:
    tool_args = locals()
    return await mcp_client.call_tool("get_price_date_tool", tool_args)

async def answer_tool_func(value: float) -> float:
    return value

max_price_tool = StructuredTool(
    name="get_max_price_dates_tool",
    func=get_max_price_dates_func,
    description="Returns the maximum price of a stock between two dates",
    args_schema=MaxPriceArgs
)
min_price_tool = StructuredTool(
    name="get_min_price_dates_tool",
    func=get_min_price_dates_func,
    description="Returns the minimum price of a stock between two dates",
    args_schema=MaxPriceArgs
)
price_tool = StructuredTool(
    name="get_price_date_tool",
    func=get_price_date_func,
    description="Returns the price of a stock on a given date",
    args_schema=PriceDateArgs
)
answer_tool = StructuredTool(
    name="answer",
    func=answer_tool_func,
    description="Answer the question",
    args_schema=AnswerArgs
)


async def main():
    await mcp_client.connect_to_server()
    url_grok = "https://5374-34-82-81-246.ngrok-free.app"
    url_grok = f"{url_grok}/generate_messages"

    colab_llm = CustomColabLLM(colab_url=url_grok)
    react_mcp_agent = ReactAgent(
            llm=colab_llm,
            tools=[price_tool, max_price_tool, min_price_tool, answer_tool],
            question="What was the top price of amazon stock during 2024-12-18 to 2025-04-07?"
            )
    messages = await react_mcp_agent.run(reset=True)

    for message in messages:
        print(message.content)
        #print("\n\n")

    await mcp_client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
