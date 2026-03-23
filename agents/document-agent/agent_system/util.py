import os
import mlflow

from langchain_openai import ChatOpenAI

def OpenAI_LLM(
        base_url: str,
        model: str = "gpt-4.1",
        temperature: float = 0.65,
        top_p: float = 0.95,
        max_completion_tokens: int = 1000,
        timeout_s: int = 20,
        max_retries: int = 1,
        seed: int = 0
    ) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout_s,
        max_retries=max_retries,
        seed=seed,
        top_p=top_p,
        # frequency_penalty=
        # streaming=True 
    )

def init_mlflow(experiment_name: str):

    assert os.environ["MLFLOW_HOST"] != "" and os.environ["MLFLOW_PORT"], "MLFLOW_HOST and MLFLOW_PORT env variables need to be set"

    print("mlflow - initializing...")
    mlflow.set_tracking_uri(f"http://{os.environ['MLFLOW_HOST']}:{os.environ['MLFLOW_PORT']}") 
    mlflow.set_experiment(experiment_name)
    print("mlflow - initialized!")

def format_system_prompt(prompt, **args):
    return {
        "role": "system",
        "content": prompt.format(**args)
    }


# async def run_mcp_client(http_client_url: str) -> AsyncGenerator[AIMessageChunk, None]:
#     async with streamable_http_client(url=http_client_url) as (read_stream, write_stream, _):
#         async with ClientSession(read_stream, write_stream) as session:
            
#             await session.initialize()

#             tools = await load_mcp_tools(session)
#             print(f"available tools: ", tools)

#             model = ChatAnthropic(
#                 model_name="claude-haiku-4-5-20251001",
#                 timeout=20.0,
#                 stop=["<STOP>"],
#                 temperature=0.35
#             )

#             agent = create_agent(
#                 model,
#                 tools
#             )

#             messages = [
#                 HumanMessage(content=["name what model you are and what tools where provided to you!"])
#             ]
            
#             async for chunk in model.astream(messages):
#                 yield chunk
            

# TODO
# def print_stream():
#     loop = asyncio.get_event_loop()
#     async def _run():
#         async for chunk in run_mcp_client(http_client_url="http://localhost:8080"):
#             print("chunk:", chunk.content)
#             # yield chunk
#     loop.run_until_complete(_run())
