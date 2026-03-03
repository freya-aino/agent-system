from pydantic import BaseModel, Field

from langgraph.graph import StateGraph
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage

from agent_system.abstract import Agent
from agent_system.tools import DocumentSearchOutput
from agent_system.util import format_system_prompt


class OA_AgentChoiceNode(BaseModel):
    notepad: str = Field(..., description="Notiere dir den zustand der Konversation um den Ist Zustand im blick zu behalten wen du entscheidungen triffst.")
    nächster_agent: str = Field(..., description="Der nächste Agent welcher ausgeführt werden soll.")

class OA_State(BaseModel):
    SelfState: List[AIMessage]
    Agents: Dict[str, Agent]

class OrchestrationAgent(Agent):
    def __init__(self, llm: ChatOpenAI, agents: List[Agent]):

        graph = StateGraph(OA_State)
        graph.add_node(self.AgentChoiceNode.__name__, self.AgentChoiceNode)
        graph.compile()

        super().__init__(
            llm=llm,
            state=OA_State(
                SelfState = [], 
                Agents = {a.__name__: a for a in agents}
            ),
            graph=graph.compile(),
            responseFormats=[
                OA_AgentChoiceNode
            ]
        )

    def AgentChoiceNode(self, state: OA_State):
        self.llm.with_structured_output(OA_AgentChoiceNode).invoke([
            format_system_prompt(
                self.prompts[OA_AgentChoiceNode],
                currentlyAvailableAgents = "---\n".join([f"# {k}\n{state.Agents[k].__doc__}\n" for k in state.Agents.keys()])
            )
        ])
        