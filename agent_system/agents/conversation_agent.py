import operator
import mlflow

from pydantic import BaseModel, Field
from typing import List, Annotated, Union
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage

from agent_system.abstract import Agent
from agent_system.util import format_system_prompt


MIN_RELEVANT_MESSAGES=0
MAX_RELEVANT_MESSAGES=4

MIN_QUESTION_PRECISION=0
MAX_QUESTION_PRECISION=9


class CA_Conversation_ResponseFormat(BaseModel):
    antwort: str = Field(..., description="Deine Antwort die die Konversation weiterführt und mehr informationen sammelt und Unklarheiten aufkößt.")
    informationen_key_points: List[str] = Field(..., description="Die Informationen oder Key-points welche du von der bisherigen Konversation erhalten kontest.")
    anzahl_relevanter_nachrichten: int = Field(..., description=f"Die N (${MIN_RELEVANT_MESSAGES}-${MAX_RELEVANT_MESSAGES}) letzten Nachrichten die relevant für das momentante Thema sind.", ge=MIN_RELEVANT_MESSAGES, le=MAX_RELEVANT_MESSAGES)

class CA_Classification_ResponseFormat(BaseModel):
    ist_frage: bool = Field(..., description="Ist eine Frage gestellt worden")
    frage_präzision: int = Field(..., description=f"Die präzision einer gestellten frage (${MIN_QUESTION_PRECISION}-${MAX_QUESTION_PRECISION})", ge=MIN_QUESTION_PRECISION, le=MAX_QUESTION_PRECISION)
    fachbereich: str = Field(..., description="Der Fachbereich der Frage.")


class CA_State(BaseModel):
    Conversation: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    InformationKeyPoints: Annotated[List[str], operator.add]
    Classification: Annotated[List[CA_Classification_ResponseFormat], operator.add]
    NumContextRelevantMessages: int

class ConversationAgent(Agent):

    def __init__(self, llm: ChatOpenAI):

        graph = StateGraph(CA_State)
        graph.add_node(self.ConversationNode.__name__, self.ConversationNode)
        graph.add_node(self.ClassificationNode.__name__, self.ClassificationNode)
        graph.add_edge(START, self.ConversationNode.__name__)
        graph.add_edge(START, self.ClassificationNode.__name__)
        graph.add_edge(self.ConversationNode.__name__, END)
        graph.add_edge(self.ClassificationNode.__name__, END)
        
        super().__init__(
            llm=llm,
            state=CA_State(
                Conversation=[],
                InformationKeyPoints=[],
                Classification=[],
                NumContextRelevantMessages=1
            ),
            graph=graph.compile(),
            responseFormats=[
                CA_Conversation_ResponseFormat,
                CA_Classification_ResponseFormat
            ],
        )

    @mlflow.trace(name="CA_ConversationNode", span_type="func")
    def ConversationNode(self, state: CA_State):

        res = self.llm.with_structured_output(CA_Conversation_ResponseFormat).invoke([
            format_system_prompt(
                self.prompts[CA_Conversation_ResponseFormat],
                allow_partial=False,
                questionairGoals=[], # TODO - read from pre-formatted txt file or smth.
                informationKeyPoints=state.InformationKeyPoints
            ),
            *state.Conversation
        ])
        res = CA_Conversation_ResponseFormat(**res)

        return {
            "Conversation": [AIMessage(content=res.antwort)],
            "InformationKeyPoints": res.informationen_key_points,
            "NumContextRelevantMessages": res.anzahl_relevanter_nachrichten,
        }

    @mlflow.trace(name="CA_ClassificationNode", span_type="func")
    def ClassificationNode(self, state: CA_State):

        res = self.llm.with_structured_output(CA_Classification_ResponseFormat).invoke([
            format_system_prompt(
                self.prompts[CA_Classification_ResponseFormat],
                allow_partial=False,
                lastUserMessage=state.Conversation[-1].content
            )
        ])
        res = CA_Classification_ResponseFormat(**res)

        return {
            "Classification": [res]
        }
