import asyncio
from langchain_core.tools import tool, Tool, StructuredTool
from client import MCPClient
from stock_agent.mcp_react_agent import CustomColabLLM, ReactAgent
from stock_agent.models import MaxPriceArgs, AnswerArgs, PriceDateArgs
from stock_agent.tools import ToolManager


async def main():
    mcp_client = MCPClient()
    await mcp_client.connect_to_server()

    tool_manager = ToolManager(mcp_client)

    url_grok = "https://5374-34-82-81-246.ngrok-free.app"
    url_grok = f"{url_grok}/generate_messages"

    colab_llm = CustomColabLLM(colab_url=url_grok)
    react_mcp_agent = ReactAgent(
            llm=colab_llm,
            tools=tool_manager.get_tool_instances(),
            question="What was the top price of amazon stock during 2024-12-18 to 2025-04-07?"
            )
    messages = await react_mcp_agent.run(reset=True)

    for message in messages:
        print(message.content)
        #print("\n\n")

    await mcp_client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
