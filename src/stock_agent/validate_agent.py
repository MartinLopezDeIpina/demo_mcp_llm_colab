import asyncio
from os.path import split

from langchain_core.tools import tool, Tool, StructuredTool
from client import MCPClient
from stock_agent.agent.mcp_react_agent import CustomColabLLM, ReactAgent
from stock_agent.agent.models import MaxPriceArgs, AnswerArgs, PriceDateArgs
from stock_agent.agent.tools import ToolManager
from datasets import load_dataset
from langchain_groq import ChatGroq
import dotenv
from langsmith import Client


async def get_agent_response(agent, question, debug):
    messages = await agent.run(question=question, reset=True)

    if debug:
        print("\n\n\n\n###### MENSAJES RECIBIDOS AGENTE VALIDACIÓN ######\n\n\n\n")
        for message in messages:
            print(message.content)
            #print("\n\n")
    return messages[-1].content, len(messages)

async def main():
    dataset_qa = load_dataset("MartinElMolon/QA_precio_stocks", split="eval")
    NUM_EVALUACIONES = 2
    """
    Messages: human, ai_thinking, action, observation, ai_thinking, action, result
    7 messgaes
    """
    NUM_PERFECT_MESSAGES = 6

    mcp_client = MCPClient()
    await mcp_client.connect_to_server()

    tool_manager = ToolManager(mcp_client)

    url_grok = "https://5495-34-142-144-179.ngrok-free.app"
    url_grok = f"{url_grok}/generate_messages"

    client = Client()

    #colab_llm = CustomColabLLM(colab_url=url_grok)
    groq_llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        verbose=True
    )
    tools = tool_manager.get_tool_instances()
    react_mcp_agent = ReactAgent(
        llm=groq_llm,
        tools=tools
    )

    datos_evaluacion = dataset_qa[:NUM_EVALUACIONES]
    questions = datos_evaluacion["question"]
    correct_answers = datos_evaluacion["price"]

    correct_responses = 0
    perfect_responses = 0
    for i in range(NUM_EVALUACIONES):
        print(f"validating {i} / {NUM_EVALUACIONES} Correct: {correct_responses}")
        try:
            agent_answer, num_messages = await get_agent_response(react_mcp_agent, questions[i], debug=False)
            # El agente puede redondear los decimales
            if float(agent_answer) - correct_answers[i] < 0.01:
                correct_responses += 1
                if num_messages == NUM_PERFECT_MESSAGES:
                    perfect_responses += 1
        # En caso de error la repsuesta será incorrecta
        except Exception as e:
            print(f"Error al ejecutar pregunta {questions[i]}: {e}")

    print(f"Percentage correct responses: {correct_responses} / {NUM_EVALUACIONES}")
    print(f"Percentage perfect responses: {perfect_responses} / {NUM_EVALUACIONES}")

    await mcp_client.cleanup()

if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
