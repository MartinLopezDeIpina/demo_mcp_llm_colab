import asyncio
from os.path import split

from langchain_core.tools import tool, Tool, StructuredTool
from client import MCPClient
from stock_agent.agent.mcp_react_agent import CustomColabLLM, ReactAgent
from stock_agent.agent.models import MaxPriceArgs, AnswerArgs, PriceDateArgs, ToolCallAction
from stock_agent.agent.tools import ToolManager
from datasets import load_dataset, Dataset, concatenate_datasets
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
    return messages[-1].content, messages

def save_batch(batch, train_dataset, remote_dataset_name, debug=False):
    new_batch_dataset = Dataset.from_list(batch)

    # first iteration will be None
    if train_dataset:
        new_dataset = concatenate_datasets([train_dataset, new_batch_dataset])
    else:
        new_dataset = new_batch_dataset

    if debug:
        print(f"##### Saving batch, new dataset length: {len(new_dataset)} ######")
    new_dataset.push_to_hub(remote_dataset_name)

    return new_dataset

def add_messages_to_batch_buffer(messages, batch_buffer, question, correct_answer):
    training_sample_messages = []
    for message in messages[:-1]:
        if type(message) != ToolCallAction:
            loss = False
            if message.type == "ai":
                loss = True
            training_sample_messages.append({
                "role": message.type,
                "content": message.content,
                "loss": loss
            })

    batch_buffer.append({
        "question": question,
        "price": correct_answer,
        "messages": training_sample_messages
    })


async def main():
    DEBUG=True
    NUM_TRAINING_DATA = 750
    INIT_INDEX=457
    """
    Messages: human, ai_thinking, action, observation, ai_thinking, action, result
    7 messgaes
    """
    NUM_PERFECT_MESSAGES = 6

    dataset_qa = load_dataset("MartinElMolon/QA_precio_stocks", split="train")
    dataset_train_qa = dataset_qa[INIT_INDEX:NUM_TRAINING_DATA]
    questions = dataset_train_qa["question"]
    correct_answers = dataset_train_qa["price"]

    REMOTE_DATASET_NAME = "MartinElMolon/stocks_demo_react_agent_generated_train_dataset"
    generated_dataset = load_dataset(REMOTE_DATASET_NAME, split="train")
    #generated_dataset = None

    BUFFER_SIZE = 10
    batch_buffer = []

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


    correct_responses = 0
    perfect_responses = 0
    for i in range(NUM_TRAINING_DATA):
        print(f"generating {i} / {NUM_TRAINING_DATA - INIT_INDEX} Correct: {correct_responses} Perfect: {perfect_responses}")
        try:
            agent_answer, messages = await get_agent_response(react_mcp_agent, questions[i], debug=False)
            num_messages = len(messages)
            # El agente puede redondear los decimales
            if float(agent_answer) - correct_answers[i] < 0.01:
                correct_responses += 1
                if num_messages == NUM_PERFECT_MESSAGES:
                    perfect_responses += 1

                    add_messages_to_batch_buffer(messages, batch_buffer, questions[i], correct_answers[i])

                    if len(batch_buffer) >= BUFFER_SIZE:
                        generated_dataset = save_batch(batch_buffer, generated_dataset, REMOTE_DATASET_NAME, debug=DEBUG)
                        batch_buffer = []

                        if not generated_dataset:
                            generated_dataset = load_dataset(REMOTE_DATASET_NAME, split="train")

        # If an error occurs, response is not correct
        except Exception as e:
            print(f"Error al ejecutar pregunta {e}")

    # if all data is processed, save the remaining data
    save_batch(batch_buffer, generated_dataset, REMOTE_DATASET_NAME, debug=DEBUG)

    print(f"Percentage correct responses: {correct_responses} / {NUM_TRAINING_DATA - INIT_INDEX}")
    print(f"Percentage perfect responses: {perfect_responses} / {NUM_TRAINING_DATA - INIT_INDEX}")

    await mcp_client.cleanup()

if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
