import os

from pydantic import BaseModel
from langchain_community.retrievers import AzureAISearchRetriever

from agent_system.abstract import DocumentRetreiver


class DocumentSearchOutput(BaseModel):
    Source: str
    Content: str


class AzureDocumentRetreiver(DocumentRetreiver):
    def __init__(self):

        #TODO - add key verification
        
        self.sourceKey = os.environ["AZURE_AI_SEARCH_SOURCE_KEY"]
        self.contentKey = os.environ["AZURE_AI_SEARCH_CONTENT_KEY"]

    def retreive(self, query: str, top_k: int):
        res = AzureAISearchRetriever(
            top_k = top_k,
            api_key=os.environ["AZURE_AI_SEARCH_API_KEY"],
            service_name=os.environ["AZURE_AI_SEARCH_SERVICE_NAME"],
            index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"],
            content_key=self.contentKey
        ).invoke(query)
        
        out = []
        for r in res:
            rr = r.model_dump()
            assert self.sourceKey in rr, f"'{self.sourceKey}' not found in vector-db return format"
            out.append(DocumentSearchOutput(Source=rr[self.sourceKey], Content=rr["metadata"][self.contentKey]))
        return out

