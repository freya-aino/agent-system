import mlflow

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Union, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage

from agent_system.abstract import Agent
from agent_system.tools import DocumentSearchOutput
from agent_system.util import format_system_prompt



class RA_ExtractCurrentQuestion(BaseModel):
    aktuelle_frage: str = Field(..., description="die aktuelle frage die der User aktiv stellt, um die es sich in der aktuellen konversation handelt.")

class RA_Reasoning(BaseModel):
    keypoints: List[str] = Field(..., description="Die wichtigsten Punkte der Informationen mit Bezug auf die User Frage.")
    gedankengang: str = Field(..., description="Die Gedanken mit welchen du eine logische rückführung formulierst, informationen über den sachverhalt kompilierst und dir eine Übersicht über die datenlage verschaffst.")

class RA_Answer(BaseModel):
    antwort: str = Field(..., description="Die Antwort auf eine informations-anfrage in welcher du dich auf den gegebenen Sachverhalt beziehst.")

class RA_Evaluate(BaseModel):
    bezug_auf_quellen: int = Field(..., description="") # TODO
    bezug_auf_sachverhalt: int = Field(..., description="") # TODO
    gedankengang_effizienz: int = Field(..., description="") # TODO


class Reasoning_Agent_State(BaseModel):
    CurrentConversation: List[Union[HumanMessage, AIMessage]]
    UserQuestion: Optional[str] = None
    DocumentChunksInContext: List[DocumentSearchOutput] = []
    Reasoning: Optional[str] = None
    Answer: Optional[str] = None

    Evaluation: Optional[RA_Evaluate] = None



class Reasoning_Agent(Agent):
    """
        Der Reasoning Agent vormuliert ein theoretischen rückshluss wie aus einer liste an Informationen eine Konkrete Antwort geschöpft werden kann.

        Dieser prozess ist kostspielig und sollte nur verwendet werden wen es unklarheiten in der Antwort gibt die mit einem Gedankenprozess gesützt werden müssen.
    """

    def __init__(self, llm: ChatOpenAI):

        graph = StateGraph(Reasoning_Agent_State)
        graph.add_node(self.ExtractUserQuestionNode.__name__, self.ExtractUserQuestionNode)
        graph.add_node(self.ReasoningNode.__name__, self.ReasoningNode)
        graph.add_node(self.AnswerNode.__name__, self.AnswerNode)
        graph.add_node(self.ValidationNode.__name__, self.ValidationNode)
        graph.add_edge(START, self.ExtractUserQuestionNode.__name__)
        graph.add_edge(self.ExtractUserQuestionNode.__name__, self.ReasoningNode.__name__)
        graph.add_edge(self.ReasoningNode.__name__, self.AnswerNode.__name__)
        graph.add_edge(self.AnswerNode.__name__, self.ValidationNode.__name__)
        graph.add_edge(self.ValidationNode.__name__, END)

        super().__init__(
            llm=llm,
            stateType=Reasoning_Agent_State,
            graph=graph.compile(),
            responseFormats=[
                RA_ExtractCurrentQuestion,
                RA_Reasoning,
                RA_Answer,
                RA_Evaluate
            ]
        )

    @mlflow.trace(name="RA_ExtractUserQuestionNode", span_type="func")
    def ExtractUserQuestionNode(self, state: Reasoning_Agent_State):
        resp = self.llm.with_structured_output(RA_ExtractCurrentQuestion).invoke([
            format_system_prompt(self.prompts[RA_ExtractCurrentQuestion]),
            *state.CurrentConversation
        ])
        resp = RA_ExtractCurrentQuestion(**resp.model_dump())
        return {
            "UserQuestion": resp.aktuelle_frage
        }

    @mlflow.trace(name="RA_ReasoningNode", span_type="func")
    def ReasoningNode(self, state: Reasoning_Agent_State):
        resp = self.llm.with_structured_output(RA_Reasoning).invoke([
            format_system_prompt(
                self.prompts[RA_Reasoning],
                userQuestion=state.UserQuestion
            )
        ])
        resp = RA_Reasoning(**resp.model_dump())
        return {
            "Reasoning": resp.gedankengang
        }

    @mlflow.trace(name="RA_AnswerNode", span_type="func")
    def AnswerNode(self, state: Reasoning_Agent_State):
        resp = self.llm.with_structured_output(RA_Answer).invoke([
            format_system_prompt(
                self.prompts[RA_Answer],
                userQuestion=state.UserQuestion,
                thoughProcess=state.Reasoning,
            )
        ])
        resp = RA_Answer(**resp.model_dump())
        return {
            "Answer": resp.antwort
        }

    @mlflow.trace(name="RA_ValidationNode", span_type="func")
    def ValidationNode(self, state: Reasoning_Agent_State):
        resp = self.llm.with_structured_output(RA_Evaluate).invoke([
            format_system_prompt(
                self.prompts[RA_Evaluate],
                userQuestion=state.UserQuestion,
                documentElements=state.DocumentChunksInContext
            ),
        ])
        resp = RA_Evaluate(**resp.model_dump())
        return {
            "Evaluation": resp
        }