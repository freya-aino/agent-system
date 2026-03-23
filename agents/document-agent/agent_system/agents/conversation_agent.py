import operator
import mlflow

from enum import Enum, auto
from pydantic import BaseModel, Field
from typing import List, Annotated, Union, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage

from agent_system.abstract import Agent
from agent_system.util import format_system_prompt


MIN_RELEVANT_MESSAGES=0
MAX_RELEVANT_MESSAGES=4

MIN_QUESTION_PRECISION=0
MAX_QUESTION_PRECISION=9


class AvailableAgents(Enum):
    CAG_Agent = auto()
    RAG_Agent = auto()
    Reasoning_Agent = auto()


class CA_Conversation(BaseModel):
    antwort: str = Field(..., description="Deine Antwort die die Konversation weiterführt und mehr informationen sammelt und Unklarheiten aufkößt.")
    informationen_key_points: List[str] = Field(..., description="Die Informationen oder Key-points welche du von der bisherigen Konversation erhalten kontest.")
    anzahl_relevanter_nachrichten: int = Field(..., description=f"Die N (${MIN_RELEVANT_MESSAGES}-${MAX_RELEVANT_MESSAGES}) letzten Nachrichten die relevant für das momentante Thema sind.", ge=MIN_RELEVANT_MESSAGES, le=MAX_RELEVANT_MESSAGES)
    note_zu_selbst: str = Field(..., description="Füge eine Persönliche einschätzung der Konversation hinzu und schreibe dir dein weiters vorgehen in einem Stichpunkt auf.")

class CA_Classification(BaseModel):
    ist_frage: bool = Field(..., description="Ist eine Frage gestellt worden")
    frage_präzision: int = Field(..., description=f"Die präzision einer gestellten frage (${MIN_QUESTION_PRECISION}-${MAX_QUESTION_PRECISION})", ge=MIN_QUESTION_PRECISION, le=MAX_QUESTION_PRECISION)
    fachbereich: str = Field(..., description="Der Fachbereich der Frage.")

class CA_ChooseAgent(BaseModel):
    gewählter_agent: AvailableAgents = Field(..., description="Welchen Agenten, falls notwendig, wälst du aus die dir Informationen beschaffen.")
    verwende_agent: bool = Field(..., description="Willst du das der ausgewählte agent verwendet wird um dier informationen zu beschaffen")
    begründung: str = Field(..., description="eine sehr kurze Begründung einen bestimmten oder keinen agenten zu wählen")

class CA_State(BaseModel):
    CurrentConversation: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    InformationKeyPoints: Annotated[List[str], operator.add] = []
    InternalState: Annotated[List[str], operator.add] = []
    Classification: Annotated[List[CA_Classification], operator.add] = []
    NumContextRelevantMessages: int = 1
    HandofAgent: Optional[CA_ChooseAgent] = None


class Conversation_Agent(Agent):

    def __init__(self, llm: ChatOpenAI, agents: List[Agent]):

        self.agents = {a.__class__.__name__: a for a in agents}

        graph = StateGraph(CA_State)
        graph.add_node(self.ConversationNode.__name__, self.ConversationNode)
        graph.add_node(self.ClassificationNode.__name__, self.ClassificationNode)
        graph.add_node(self.AgentChoiceNode.__name__, self.AgentChoiceNode)
        graph.add_edge(START, self.ConversationNode.__name__)
        graph.add_edge(START, self.ClassificationNode.__name__)
        graph.add_edge(self.ConversationNode.__name__, self.AgentChoiceNode.__name__)
        graph.add_edge(self.ClassificationNode.__name__, self.AgentChoiceNode.__name__)
        graph.add_edge(self.AgentChoiceNode.__name__, END)
        
        super().__init__(
            llm=llm,
            stateType=CA_State,
            graph=graph.compile(),
            responseFormats=[
                CA_Conversation,
                CA_Classification,
                CA_ChooseAgent
            ],
        )

    @mlflow.trace(name="CA_Conversation", span_type="func")
    def ConversationNode(self, state: CA_State):

        res = self.llm.with_structured_output(CA_Conversation).invoke([
            format_system_prompt(
                self.prompts[CA_Conversation],
                questionairGoals=[], # TODO - read from pre-formatted txt file or smth.
                informationKeyPoints=state.InformationKeyPoints
            ),
            *state.CurrentConversation
        ])
        res = CA_Conversation(**res.model_dump())

        return {
            "CurrentConversation": [AIMessage(content=res.antwort)],
            "InternalState": [res.note_zu_selbst],
            "InformationKeyPoints": res.informationen_key_points,
            "NumContextRelevantMessages": res.anzahl_relevanter_nachrichten,
        }

    @mlflow.trace(name="CA_Classification", span_type="func")
    def ClassificationNode(self, state: CA_State):

        res = self.llm.with_structured_output(CA_Classification).invoke([
            format_system_prompt(
                self.prompts[CA_Classification],
                lastUserMessage=state.CurrentConversation[-1].content
            )
        ])
        res = CA_Classification(**res.model_dump())
        return {
            "Classification": [res]
        }

    @mlflow.trace(name="CA_AgentChoice", span_type="func")
    def AgentChoiceNode(self, state: CA_State):
        resp = self.llm.with_structured_output(CA_ChooseAgent).invoke([
            format_system_prompt(
                self.prompts[CA_ChooseAgent],
                currentlyAvailableAgents = [f"# {type(a).__name__}\n{a.__doc__}\n" for a in self.agents.values()]
            ),
            *state.InternalState
        ])
        resp = CA_ChooseAgent(**resp.model_dump())

        return {
            "HandofAgent": resp
        }
        