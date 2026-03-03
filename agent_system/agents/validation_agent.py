from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from agent_system.abstract import Agent
from agent_system.tools import DocumentSearchOutput
from agent_system.util import format_system_prompt



class VA_Evaluate_ResponseFormat(BaseModel):
    bezug_auf_quellen: int = Field(..., description="") # TODO
    bezug_auf_sachverhalt: int = Field(..., description="") # TODO
    gedankengang_effizienz: int = Field(..., description="") # TODO

class VA_State(BaseModel):
    UserQuestion: str
    DocumentChunks: List[DocumentSearchOutput]
    Reasoning: str
    Answer: str
    Evaluation: Optional[VA_Evaluate_ResponseFormat]

    
class ValidationAgent(Agent):
    def __init__(self, llm: ChatOpenAI):

        graph = StateGraph(VA_State)
        graph.add_node(self.ValidationNode.__name__, self.ValidationNode)
        graph.add_edge(START, self.ValidationNode.__name__)
        graph.add_edge(self.ValidationNode.__name__, END)

        super().__init__(
            llm=llm,
            state=VA_State(
                UserQuestion="",
                DocumentChunks=[],
                Reasoning="",
                Answer="",
                Evaluation=None
            ),
            graph=graph.compile(),
            responseFormats=[
                VA_Evaluate_ResponseFormat
            ]
        )

    def ValidationNode(self, state: VA_State):
        resp = self.llm.with_structured_output(VA_Evaluate_ResponseFormat).invoke([
            format_system_prompt(
                self.prompts[VA_Evaluate_ResponseFormat],
                userQuestion=state.UserQuestion,
                documentElements=state.DocumentChunks
            ),
        ])
        resp = VA_Evaluate_ResponseFormat(**resp)
        return {
            "Evaluation": resp
        }