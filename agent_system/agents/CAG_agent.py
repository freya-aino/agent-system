import mlflow

from pydantic import BaseModel, Field
from typing import List, Union, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage

from agent_system.abstract import Agent
from agent_system.util import format_system_prompt


class CAG_Agent_DomainClassification(BaseModel):
    frage_ist_in_domänenwissen: bool = Field(..., description="") # TODO
    relevante_domänen_indexe: List[int] = Field(..., description="") # TODO - restrict to max number of references and restrict to length of domain knowledge
    
class CAG_Agent_Answer(BaseModel):
    rechtfertigung_aus_kontext: str = Field(..., description="") # TODO
    antwort: str = Field(..., description="") # TODO

class CAG_Agent_State(BaseModel):
    CurrentConversation: List[Union[HumanMessage, AIMessage]]
    Classification: Optional[CAG_Agent_DomainClassification] = None
    DomainKnowledge: List[str] = []
    Answer: Optional[CAG_Agent_Answer] = None


class CAG_Agent(Agent):
    """
        Ein Agent der mittels von einem "Context Aware Generation" mechanismus auf Domänen spezifischen Wissen eine Aussage Trifft.

        Diese Aussage ist Genauer als eine Kontext lose Aussage.
        Die Aussage ist auch günstiger als andere arten der Kontext generation, aber dafür nicht so präzise.
    """


    def __init__(self, llm: ChatOpenAI):

        graph = StateGraph(CAG_Agent_State)
        graph.add_node(self.DomainKnowledgeClassificationNode.__name__, self.DomainKnowledgeClassificationNode)
        graph.add_node(self.AnswerFromDomainKnowledgeNode.__name__, self.AnswerFromDomainKnowledgeNode)
        graph.add_node(self.IsDomainRelevantNode.__name__, self.IsDomainRelevantNode)
        graph.add_edge(START, self.DomainKnowledgeClassificationNode.__name__)
        graph.add_conditional_edges(
            self.DomainKnowledgeClassificationNode.__name__,
            self.IsDomainRelevantNode,
            {
                True: self.AnswerFromDomainKnowledgeNode.__name__,
                False: END
            }
        )
        graph.add_edge(self.AnswerFromDomainKnowledgeNode.__name__, END)

        super().__init__(
            llm=llm,
            stateType=CAG_Agent_State,
            graph=graph.compile(),
            responseFormats=[
                CAG_Agent_DomainClassification,
                CAG_Agent_Answer
            ]
        )
    
    @mlflow.trace(name="DA_IsDomainRelevantNode", span_type="func")
    def IsDomainRelevantNode(self, state: CAG_Agent_State):
        assert state.Classification, "Classification should never be none at this point!"
        if not state.Classification.frage_ist_in_domänenwissen:
            return False
        if len(state.Classification.relevante_domänen_indexe) <= 0:
            return False
        return True
    
    @mlflow.trace(name="DA_DomainKnowledgeClassificationNode", span_type="func")
    def DomainKnowledgeClassificationNode(self, state: CAG_Agent_State):
        resp = self.llm.with_structured_output(CAG_Agent_DomainClassification).invoke([
            format_system_prompt(
                self.prompts[CAG_Agent_DomainClassification],
                domainKnowledge=state.DomainKnowledge
            ),
            *state.CurrentConversation
        ])
        resp = CAG_Agent_DomainClassification(**resp.model_dump())
        return {
            "Classification": resp
        }
        
    @mlflow.trace(name="DA_AnswerFromDomainKnowledgeNode", span_type="func")
    def AnswerFromDomainKnowledgeNode(self, state: CAG_Agent_State):
        assert state.Classification, "Classification should never be none at this point!"
        assert state.Classification.relevante_domänen_indexe, "Viable domain indecies where never provided by the model, this should be caught earlier"

        resp = self.llm.with_structured_output(CAG_Agent_Answer).invoke([
            format_system_prompt(
                self.prompts[CAG_Agent_Answer], 
                relevantDomainKnowledge=[state.DomainKnowledge[i] for i in state.Classification.relevante_domänen_indexe]
            ),
            *state.CurrentConversation
        ])
        state.Answer = CAG_Agent_Answer(**resp.model_dump())

        return state
