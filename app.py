import dotenv
import os
import mlflow
import mlflow.openai as mlfopenai
import mlflow.langchain as mlflangchain

from typing import Union, List
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool

from agent_system import agents
from agent_system.util import init_mlflow
from agent_system.agents.conversation_agent import AvailableAgents

from typing import List
import asyncio, json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_system import agents
from dotenv import load_dotenv

# --- Modelle ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

app = FastAPI()

#import mlflow

# Calling autolog for LangChain will enable trace logging.
#mlflow.langchain.autolog()

# Optional: Set a tracking URI and an experiment
#mlflow.set_experiment("Timsturbotest")
#mlflow.set_tracking_uri("http://mlflow:10008")


mlfopenai.autolog()
# mlflangchain.autolog()

dotenv.load_dotenv()

init_mlflow(experiment_name="test-experiment")

from agent_system.util import OpenAI_LLM
llm = OpenAI_LLM(os.environ["OPENAI_ENDPOINT"])


@app.post("/v1/chat/completions")
async def completions(req: ChatRequest):
    # (Optional) .env nur einmal in app startup laden, nicht bei jeder Anfrage
    load_dotenv()

    # Imports hier nur der Übersicht halber – besser: Modul-Top (Performance)
    from langchain_core.messages import HumanMessage, AIMessage

    # Queue für Progress-Events
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Progress-Callback: thread-safe (hier nicht nötig, aber robust)
    def on_progress(evt: dict):
        loop.call_soon_threadsafe(queue.put_nowait, evt)


    conversation_agent = agents.Conversation_Agent(llm=llm, agents=[])
    ca_state = agents.CA_State(CurrentConversation = [m for m in req.messages])


    async def producer():
        
        # Starte den Agenten asynchron im gleichen Event-Loop
        task = asyncio.create_task(conversation_agent.graph.ainvoke(ca_state))

        try:
            # Streame solange Progress, bis der Task fertig ist
            while True:
                if queue.empty():
                    if task.done():
                        break
                    await asyncio.sleep(0.05)
                else:
                    evt = await queue.get()
                    yield (json.dumps(evt) + "\n").encode("utf-8")

            # Finale Antwort
            result = await task

            if result['lastNodeKonversation']:
                print(result['konversation'][-1].content)
                result = result['konversation'][-1].content
            else:
                print(result['konversation'][-1].content)
                result = result['gedankengang']['antwort'].content

            print(result)
            yield (json.dumps({"type": "result", "data": result}) + "\n").encode("utf-8")

        except Exception as e:
            yield (json.dumps({"type": "error", "message": str(e)}) + "\n").encode("utf-8")
            raise

    return StreamingResponse(
        producer(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "yes",  # NGINX: Buffering aus
        },
    )