import asyncio
from os.path import split

from langchain_core.tools import tool, Tool, StructuredTool
from client import MCPClient
from stock_agent.mcp_react_agent import CustomColabLLM, ReactAgent
from stock_agent.models import MaxPriceArgs, AnswerArgs, PriceDateArgs
from stock_agent.tools import ToolManager
from datasets import load_dataset

async def get_agent_response(agent, question, debug):
    messages = await agent.run(question=question, reset=True)

    if debug:
        print("\n\n\n\n###### MENSAJES RECIBIDOS AGENTE VALIDACIÓN ######\n\n\n\n")
        for message in messages:
            print(message.content)
            #print("\n\n")
    return messages[-1].content

async def main():
    dataset_qa = load_dataset("MartinElMolon/QA_precio_stocks", split="eval")
    NUM_EVALUACIONES = 2

    mcp_client = MCPClient()
    await mcp_client.connect_to_server()

    tool_manager = ToolManager(mcp_client)

    url_grok = "https://141b-34-16-223-19.ngrok-free.app"
    url_grok = f"{url_grok}/generate_messages"

    colab_llm = CustomColabLLM(colab_url=url_grok)
    tools = tool_manager.get_tool_instances()
    react_mcp_agent = ReactAgent(
        llm=colab_llm,
        tools=tools
    )


    datos_evaluacion = dataset_qa[:NUM_EVALUACIONES]
    questions = datos_evaluacion["question"]
    correct_answers = datos_evaluacion["price"]

    correct_responses = 0
    for i in range(NUM_EVALUACIONES):
        try:
            agent_answer = await get_agent_response(react_mcp_agent, questions[i], debug=True)
            # El agente puede redondear los decimales
            if float(agent_answer) - correct_answers[i] < 0.01:
                correct_responses += 1
        # En caso de error la repsuesta será incorrecta
        except Exception as e:
            print(f"Error al ejecutar pregunta {questions[i]}: {e}")

    print(f"Porcentaje de respuestas correctas: {correct_responses} / {NUM_EVALUACIONES}")

    await mcp_client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
