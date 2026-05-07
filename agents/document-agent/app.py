import dotenv
import os
import mlflow
import mlflow.openai as mlfopenai
import mlflow.langchain as mlflangchain
import asyncio, json

from contextlib import asynccontextmanager

from typing import Dict
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool

from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_system import agents
from agent_system.util import init_mlflow
from agent_system.agents.conversation_agent import AvailableAgents

from agent_system import agents


# (Optional) .env nur einmal in app startup laden, nicht bei jeder Anfrage
from dotenv import load_dotenv
load_dotenv()


# --- Modelle ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    conversation_uuid: str # TODO - make this a standarized hash
    current_message: Message


#import mlflow

# Calling autolog for LangChain will enable trace logging.
#mlflow.langchain.autolog()

# Optional: Set a tracking URI and an experiment
#mlflow.set_experiment("Timsturbotest")
#mlflow.set_tracking_uri("http://mlflow:10008")

# ---- INIT ----
mlfopenai.autolog()
dotenv.load_dotenv()
conversation_state: Dict[str, agents.Agent] = {}



# TODO - placeholder for loading conversations from some persistent cache
def load_conversations() -> Dict[str, agents.Agent]:

    from agent_system.util import OpenAI_LLM
    llm = OpenAI_LLM(os.environ["OPENAI_ENDPOINT"])

    return {
        "default": agents.Conversation_Agent(llm=llm, agents=[
            agents.CAG_Agent(llm=llm), 
            agents.RAG_Agent(llm=llm), 
            agents.Reasoning_Agent(llm=llm)
        ])
    }

@asynccontextmanager
async def lifespan(app: FastAPI):

    init_mlflow(experiment_name=f"experiment (app-version={app.version})")
    conversation_state = load_conversations()

    yield
    conversation_state.clear()


app = FastAPI(
    lifespan=lifespan, 
    version="0.1.0"
)


@app.post("/v1/chat/completion")
async def completion(req: ChatRequest):

    # TODO - placeholder for managing client uuid for conversation caching
    if not req.conversation_uuid:
        agent = conversation_state["default"]
    else:
        agent = conversation_state[req.conversation_uuid]

    
    async def producer():

        stream_return = agent.aforward(**agent.state.model_dump())

        yield (stream_return)
        
        # # Queue für Progress-Events
        # queue: asyncio.Queue = asyncio.Queue()

        # try:
        #     # Streame solange Progress, bis der Task fertig ist
        #     while True:
        #         if queue.empty():
        #             if task.done():
        #                 break
        #             await asyncio.sleep(0.05)
        #         else:
        #             evt = await queue.get()
        #             yield (json.dumps(evt) + "\n").encode("utf-8")

        #     # Finale Antwort
        #     result = await task

        #     if result['lastNodeKonversation']:
        #         print(result['konversation'][-1].content)
        #         result = result['konversation'][-1].content
        #     else:
        #         print(result['konversation'][-1].content)
        #         result = result['gedankengang']['antwort'].content

        #     print(result)
        #     yield (json.dumps({"type": "result", "data": result}) + "\n").encode("utf-8")

        # except Exception as e:
        #     yield (json.dumps({"type": "error", "message": str(e)}) + "\n").encode("utf-8")
        #     raise


    return StreamingResponse(
        producer(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "yes",  # NGINX: Buffering aus
        },
    )