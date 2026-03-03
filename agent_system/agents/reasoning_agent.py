from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Union, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage

from agent_system.abstract import Agent
from agent_system.tools import DocumentSearchOutput
from agent_system.util import format_system_prompt



class RA_ExtractCurrentQuestion_ResponseFormat(BaseModel):
    aktuelle_frage: str = Field(..., description="die aktuelle frage die der User aktiv stellt, um die es sich in der aktuellen konversation handelt.")

class RA_Reasoning_ResponseFormat(BaseModel):
    keypoints: List[str] = Field(..., description="Die wichtigsten Punkte der Informationen mit Bezug auf die User Frage.")
    gedankengang: str = Field(..., description="Die Gedanken mit welchen du eine logische rückführung formulierst, informationen über den sachverhalt kompilierst und dir eine Übersicht über die datenlage verschaffst.")

class RA_Answer_ResponseFormat(BaseModel):
    antwort: str = Field(..., description="Die Antwort auf eine informations-anfrage in welcher du dich auf den gegebenen Sachverhalt beziehst.")

class RA_State(BaseModel):
    CurrentConversation: List[Union[HumanMessage, AIMessage]]
    UserQuestion: Optional[str]
    DocumentChunksInContext: List[DocumentSearchOutput]
    Reasoning: Optional[str]
    Answer: Optional[str]


class ReasoningAgent(Agent):
    def __init__(self, llm: ChatOpenAI):

        graph = StateGraph(RA_State)
        graph.add_node(self.ExtractUserQuestionNode.__name__, self.ExtractUserQuestionNode)
        graph.add_node(self.ReasoningNode.__name__, self.ReasoningNode)
        graph.add_node(self.AnswerNode.__name__, self.AnswerNode)
        graph.add_edge(START, self.ExtractUserQuestionNode.__name__)
        graph.add_edge(self.ExtractUserQuestionNode.__name__, self.ReasoningNode.__name__)
        graph.add_edge(self.ReasoningNode.__name__, self.AnswerNode.__name__)
        graph.add_edge(self.AnswerNode.__name__, END)

        super().__init__(
            llm=llm,
            state=RA_State(
                CurrentConversation=[],
                DocumentChunksInContext=[],
                UserQuestion=None,
                Reasoning=None,
                Answer=None
            ),
            graph=graph.compile(),
            responseFormats=[
                RA_ExtractCurrentQuestion_ResponseFormat,
                RA_Reasoning_ResponseFormat,
                RA_Answer_ResponseFormat
            ]
        )

    def ExtractUserQuestionNode(self, state: RA_State):
        resp = self.llm.with_structured_output(RA_ExtractCurrentQuestion_ResponseFormat).invoke([
            format_system_prompt(self.prompts[RA_ExtractCurrentQuestion_ResponseFormat]),
            *state.CurrentConversation
        ])
        resp = RA_ExtractCurrentQuestion_ResponseFormat(**resp)
        return {
            "UserQuestion": resp.aktuelle_frage
        }

    def ReasoningNode(self, state: RA_State):
        resp = self.llm.with_structured_output(RA_Reasoning_ResponseFormat).invoke([
            format_system_prompt(
                self.prompts[RA_Reasoning_ResponseFormat],
                userQuestion=state.UserQuestion
            )
        ])
        resp = RA_Reasoning_ResponseFormat(**resp)
        return {
            "Reasoning": resp.gedankengang
        }

    def AnswerNode(self, state: RA_State):
        resp = self.llm.with_structured_output(RA_Answer_ResponseFormat).invoke([
            format_system_prompt(
                self.prompts[RA_Answer_ResponseFormat],
                userQuestion=state.UserQuestion,
                thoughProcess=state.Reasoning,
            )
        ])
        resp = RA_Answer_ResponseFormat(**resp)
        return {
            "Answer": resp.antwort
        }
