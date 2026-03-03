import os

from mlflow import genai
from abc import ABC
from typing import List
from pydantic import BaseModel
from enum import Enum

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph


# def initializePrompts(self, prompt_dir: str, language: str = "de"): # TODO - dont hardcode language

#     assert os.path.isdir(prompt_dir), f"path for prompts '{prompt_dir}' does not exist!"

#     for respFormat in self.responseFormats:

#         with open(f"{prompt_dir}/{respFormat.__name__}.md") as f:
#             template = f.read()

#         genai.register_prompt(
#             name=respFormat.__name__,
#             template=template,
#             response_format=respFormat,
#             commit_message="Initial commit",
#             tags={
#                 "language": language,
#             },
#         )

def initializePrompt(promptsDir: str, responseFormat: type[BaseModel], language: str = "de"): # TODO - dont hardcode language
    
    assert os.path.isdir(promptsDir), f"path for prompts '{promptsDir}' does not exist!"

    with open(f"{promptsDir}/{responseFormat.__name__}.md") as f:
        template = f.read()
    
    genai.register_prompt(
        name=responseFormat.__name__,
        template=template,
        response_format=responseFormat,
        commit_message="Initial commit",
        tags={
            "language": language,
        },
    )


class Language(Enum):
    DE = "de"

class Agent(ABC):
    def __init__(self, llm: ChatOpenAI, state: BaseModel, graph: CompiledStateGraph, responseFormats: List[type[BaseModel]], promptsDir: str = "./agent_system/prompts"):
        # self.responseFormats = responseFormats
        self.llm = llm # TODO - make llm configurable for each step
        self.graph = graph
        self.state = state

        # TODO - add prompt versioning
        version = "latest"
        self.prompts = {}
        for rf in responseFormats:
            p = genai.search_prompts(f"name='{rf.__name__}'")
            if len(p) <= 0:
                print(f"no prompts found with the name='{rf.__name__}', creating ...")
                initializePrompt(promptsDir=promptsDir, responseFormat=rf)
            
            print(f"prompts found for name='{rf.__name__}', loading...")
            self.prompts[rf] = genai.load_prompt(f"prompts:/{rf.__name__}@{version}")

    def forward(self):
        return type(self.state)(**self.graph.invoke(self.state))
    
    def update(self, newState: BaseModel):
        self.state = newState


class DocumentRetreiver(ABC):
    def __init__(self):
        raise NotImplementedError

    def retreive(self, query: str, top_k: int):
        raise NotImplementedError

