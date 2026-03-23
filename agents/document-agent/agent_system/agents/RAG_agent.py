import mlflow

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Union, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage

from agent_system.abstract import Agent
from agent_system.util import format_system_prompt
from agent_system.tools import AzureDocumentRetreiver, DocumentSearchOutput


MIN_DOCUMENT_CHUNKS_RETREIVED = 1
MAX_DOCUMENT_CHUNKS_RETREIVED= 3


class RAG_Agent_RefinedSearchQueries(BaseModel):
    keyword_suche: str = Field(..., description="Die such Query des users wird umformuliert so das sie auschließlich Informationen zu den verwendeten keywords in der Wissensdatenbank sucht.")
    kontext_suche: str = Field(..., description="Die such Query des users wird umformuliert so das sie im kontext der Frage relevante Informationen aus der Wissensdatenbank sucht.")
    auser_kontext_suche: str = Field(..., description="Die such Query des users wird umformuliert so das sie auserhalb des kontexts der Frage, mit sekundärem bezug auf den Kontext, Informationen aus der Wissensdatenbank sucht. ")
    anzahl_dokument_elemente: int = Field(..., description=f"Die Anzahl (${MIN_DOCUMENT_CHUNKS_RETREIVED}-${MAX_DOCUMENT_CHUNKS_RETREIVED}) an Dokument elementen welche aus der Wissensdatenbank gehohlt werden.", ge=MIN_DOCUMENT_CHUNKS_RETREIVED, le=MAX_DOCUMENT_CHUNKS_RETREIVED)

class RAG_Agent_State(BaseModel):
    CurrentConversation: List[Union[HumanMessage, AIMessage]]
    RefinedQueries: Optional[RAG_Agent_RefinedSearchQueries] = None
    DocumentChunksInContext: list[DocumentSearchOutput] = []


class RAG_Agent(Agent):
    """
        Der Agent verwendet "Retreival Augmented Generation" um mit einer liste an Fragen wichtige Dokument elemente zu finden und diese in den Kontext zu laden.

        Diese art der Generation is teuer, führt aber zu der höchsten genauigkeit an informationen 
    """

    def __init__(self, llm: ChatOpenAI):

        graph = StateGraph(RAG_Agent_State)
        graph.add_node(self.SearchQueryRefinementNode.__name__, self.SearchQueryRefinementNode)
        graph.add_node(self.DocumentSearchNode.__name__, self.DocumentSearchNode)
        graph.add_edge(START, self.SearchQueryRefinementNode.__name__)
        graph.add_edge(self.SearchQueryRefinementNode.__name__, self.DocumentSearchNode.__name__)
        graph.add_edge(self.DocumentSearchNode.__name__, END)

        super().__init__(
            llm=llm,
            stateType=RAG_Agent_State,
            # state=RAG_Agent_State(
            #     CurrentConversation=[],
            #     RefinedQueries=None,
            #     DocumentChunksInContext=[]
            # ),
            graph=graph.compile(),
            responseFormats=[
                RAG_Agent_RefinedSearchQueries
            ]
        )
    
    @mlflow.trace(name="KA_SearchQueryRefinementNode", span_type="func")
    def SearchQueryRefinementNode(self, state: RAG_Agent_State):
        resp = self.llm.with_structured_output(RAG_Agent_RefinedSearchQueries).invoke([
            format_system_prompt(
                self.prompts[RAG_Agent_RefinedSearchQueries], 
                documentSystemKnowledge=[]  # TODO add document system knowledge
            ),
            *state.CurrentConversation
        ])
        resp = RAG_Agent_RefinedSearchQueries(**resp.model_dump())
        return {
            "RefinedQueries": resp
        }

    @mlflow.trace(name="KA_DocumentSearchNode", span_type="func")
    def DocumentSearchNode(self, state: RAG_Agent_State):

        assert state.RefinedQueries, "'RefinedQueries' should always be set when this is called!"

        top_k = min(MAX_DOCUMENT_CHUNKS_RETREIVED, max(MIN_DOCUMENT_CHUNKS_RETREIVED, state.RefinedQueries.anzahl_dokument_elemente))

        # TODO - dont hardcode document retreiver type
        chunks = AzureDocumentRetreiver().retreive(query=state.RefinedQueries.auser_kontext_suche, top_k=top_k)

        return {
            "DocumentChunksInContext": chunks
        }