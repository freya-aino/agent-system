import os

from enum import Enum
from mlflow import genai
from abc import ABC
from typing import List
from pydantic import BaseModel

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

def getLocalPromptTemplate(promptsDir: str, name: str):
    assert os.path.isdir(promptsDir), f"path for prompts '{promptsDir}' does not exist!"
    with open(f"{promptsDir}/{name}.md") as f:
        template = f.read()
    return template
    
def uploadPromptTemplate(template: str, responseFormat: type[BaseModel], language: str = "de"): # TODO - dont hardcode language
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
    def __init__(self, llm: ChatOpenAI, stateType: type[BaseModel], graph: CompiledStateGraph, responseFormats: List[type[BaseModel]], promptsDir: str = "./agent_system/prompts"):
        # self.responseFormats = responseFormats
        self.llm = llm # TODO - make llm configurable for each step
        self.graph = graph
        # self.state = type(self.state)(**state.model_dump())
        self.stateType = stateType

        # TODO - add prompt versioning and add new prompt version
        version = "latest"
        self.prompts = {}
        for rf in responseFormats:

            pp = genai.search_prompts(f"name='{rf.__name__}'")
            local_p = getLocalPromptTemplate(promptsDir=promptsDir, name=rf.__name__)

            # create new if non exist
            if len(pp) <= 0:
                print(f"no prompts found with the name='{rf.__name__}', creating ...")
                uploadPromptTemplate(template=local_p, responseFormat=rf)
            # else:
            #     p = genai.load_prompt(rf.__name__)
            #     # make new version if local template changed
            #     if p.template != local_p:
            #         print(f"new version for prompt name='{rf.__name__}' found, updating...")
            #         uploadPromptTemplate(template=local_p, responseFormat=rf)

            print(f"loading prompt '{rf.__name__}' ...")
            p = genai.load_prompt(rf.__name__)
            self.prompts[rf] = p

    def forward(self, **kwargs):
        state = self.stateType(**kwargs)
        return self.graph.invoke(state)


class DocumentRetreiver(ABC):
    def __init__(self):
        raise NotImplementedError

    def retreive(self, query: str, top_k: int):
        raise NotImplementedError
