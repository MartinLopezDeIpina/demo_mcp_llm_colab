import asyncio
import pdb
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from anthropic import Anthropic

SERVER_IP="68.219.187.57"
SERVER_PORT=8000

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self):

      print("Connecting to server...")

      self.session = await self.exit_stack.enter_async_context(
         sse_client(f"http://{SERVER_IP}:{SERVER_PORT}/sse")
         )

      streams = await self.exit_stack.enter_async_context(
          sse_client(f"http://{SERVER_IP}:{SERVER_PORT}/sse")
      )
       
      self.session = await self.exit_stack.enter_async_context(
          ClientSession(streams[0], streams[1])
      )
       
      await self.session.initialize()
      print("Session initialized")

      response = await self.session.list_tools()
      print(f"respuesta: {response}")
      tools = response.tools
      print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
      await self.exit_stack.aclose()

    async def call_tool(self, tool_name, tool_args):

      result = await self.session.call_tool(tool_name, tool_args)

      return result


async def main():

    client = MCPClient()
    await client.connect_to_server()

    try:
        await client.connect_to_server()

        tool_name = "get_max_price_dates_tool"
        tool_args = {
          "stock":"sp500",
          "start_date":"2024-06-06",
          "end_date":"2025-06-10"
        }

        result = await client.call_tool(tool_name, tool_args)
        print(result)

    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())


