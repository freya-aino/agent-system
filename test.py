from typing import List
import asyncio, json, uuid, polars as pl
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


@app.post("/v1/chat/completions")
async def completions(req: ChatRequest):
    # (Optional) .env nur einmal in app startup laden, nicht bei jeder Anfrage
    load_dotenv()

    # Imports hier nur der Übersicht halber – besser: Modul-Top (Performance)
    from agents.knowledge import WissensAgent, WissensAgentState
    from langchain_core.messages import HumanMessage, AIMessage

    # Beispiele laden
    df = pl.read_excel("./beispiele.xlsx")
    beispiele = [
        {
            "frage": r['orginal_frage'],
            "gedanken": r['gedanken'],
            "antwort": r['antwort'],
            "bewertung": {
                "bezug_auf_quellen": str(r['bewertung_bezug_auf_quellen']),
                "bezug_auf_sachverhalt": str(r['bewertung_bezug_auf_sachverhalt']),
                "gedankengang_effizienz": str(r['bewertung_gedankengang_effizienz']),
            }
        }
        for r in df.iter_rows(named=True)
    ]

    # Queue für Progress-Events
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Progress-Callback: thread-safe (hier nicht nötig, aber robust)
    def on_progress(evt: dict):
        loop.call_soon_threadsafe(queue.put_nowait, evt)

    # Agent vorbereiten
    agent = WissensAgent(
        max_llm_calls=6,
        erwuenschte_note=3.0,
        beste_note=1.4,
        on_progress=on_progress
    ).compile()

    # Konversation aus Messages bauen
    konv = []
    for m in req.messages:
        if m.role == 'user':
            konv.append(HumanMessage(content=m.content))
        else:
            konv.append(AIMessage(content=m.content))

    state = WissensAgentState(
        konversation=konv,
        klassifikation=None,
        llm_calls=0,
        dokument_elemente_in_kontext=[],
        gedankengang=None,
        beispiele=beispiele,
        lastNodeKonversation=False
    )

    async def producer():
        # Starte den Agenten asynchron im gleichen Event-Loop
        task = asyncio.create_task(agent.ainvoke(state))  # <<<< wichtig: ainvoke

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