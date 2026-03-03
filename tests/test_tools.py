import os

from agent_system.tools import AzureDocumentRetreiver

def test_azure_document_retreiver_key_prereq():
    assert os.environ["AZURE_AI_SEARCH_API_KEY"], "'AZURE_AI_SEARCH_API_KEY' environment variable not set."
    assert os.environ["AZURE_AI_SEARCH_INDEX_NAME"], "'AZURE_AI_SEARCH_INDEX_NAME' environment variable not set."
    assert os.environ["AZURE_AI_SEARCH_SERVICE_NAME"], "'AZURE_AI_SEARCH_SERVICE_NAME' environment variable not set."
    assert os.environ["AZURE_AI_SEARCH_SOURCE_KEY"], "'AZURE_AI_SEARCH_SOURCE_KEY' environment variable not set."
    assert os.environ["AZURE_AI_SEARCH_CONTENT_KEY"], "'AZURE_AI_SEARCH_CONTENT_KEY' environment variable not set."


def test_azure_document_retreiver():
    AzureDocumentRetreiver().retreive(query="test query", top_k=1)