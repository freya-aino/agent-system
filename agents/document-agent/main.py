import json
import asyncio
import dotenv
import os
import mlflow
import mlflow.openai as mlfopenai
import mlflow.langchain as mlflangchain

from typing import Union, List
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool

from agent_system import agents
from agent_system.util import init_mlflow
from agent_system.agents.conversation_agent import AvailableAgents

# TODO - add prompt version
# TODO - add production prompt # prompt = mlflow.genai.load_prompt("prompts:/my_prompt@production")
# TODO - generalize the tool calls to using tool calling strategy so that models that dont have native structured output can also use tools
# class ProductReview(BaseModel):
#     """Analysis of a product review."""
#     rating: int | None = Field(description="The rating of the product", ge=1, le=5)
#     sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
#     key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")


async def main():
    
    # TODO - manage secrets outside
    dotenv.load_dotenv()

    # init_mlflow(experiment_name="test-experiment")
    # mlfopenai.autolog()
    # mlflangchain.autolog()

    from agent_system.util import OpenAI_LLM
    llm = OpenAI_LLM(os.environ["OPENAI_ENDPOINT"])

    conversation_agent = agents.Conversation_Agent(llm=llm, agents=[])
    state = agents.CA_State(CurrentConversation=[HumanMessage("Hi kasnnt du mir eine kurze zusammenfassung von einer übersicht über Spechte geben, schreibe 2 Paragraphen")])

    outputs = {}
    conversation_bracket_count = 0
    async for chunk, context in conversation_agent.aforward(**state.model_dump()):

        if context["langgraph_node"] in outputs:
            outputs[context["langgraph_node"]] += chunk.content
        else:
            outputs[context["langgraph_node"]] = chunk.content

        if context["langgraph_node"] != "ConversationNode":
            continue

        if "{" in chunk.content:
            conversation_bracket_count += 1
        if "}" in chunk.content:
            conversation_bracket_count -= 1

        if conversation_bracket_count == 0:
            # print(chunk.content, end="", flush=True)
            print(chunk.content)


        # if "{" in chunk.content:
        #     is_currently_structured_output = True
        #     new_structured_output = True
        # if "}" in chunk.content:
        #     is_currently_structured_output = False

        # if new_structured_output:
        #     new_structured_output = False
        #     structured_outputs.append([""])
        # if is_currently_structured_output:
        #     structured_outputs[-1] += chunk.content
        # else:
        #     normal_output += chunk.content
        #     print(chunk.content, end="", flush=True)

if __name__ == "__main__":

    asyncio.run(main())
    


    # ca_state = agents.CA_State(**
    #     {
    #         "CurrentConversation": [HumanMessage("")]
    #     }
    # )

    # agent_states = {k: None for k, a in conversation_agent.agents.items()}


    # for _ in range(10):

    #     ca_state = agents.CA_State(**ca_state.model_dump())
    #     ca_state = agents.CA_State(**conversation_agent.forward(**ca_state.model_dump()))

    #     print(f"answer     => {ca_state.CurrentConversation[-1].content}")


    #     # print(f"next_agent => {ca_state.HandofAgent}")

    #     # agent_name = ca_state.HandofAgent.gewählter_agent.name

    #     # # if ca_state.HandofAgent.verwende_agent:

    #     # next_agent = conversation_agent.agents[agent_name]
    #     # current_agent_state = agent_states[agent_name]
    #     # current_agent_state.CurrentConversation.append(ca_state.CurrentConversation[:-2])


    #     # print(current_agent_state.model_dump())

    #     # new_agent_state = next_agent.forward(**current_agent_state)
    #     # agent_states[ca_state.HandofAgent.name] = new_agent_state

    #     # print(f"Agent out  => {next_agent}")




    # da_state = domain_agent.forward(**ca_state)
    # ka_state = knowledge_agent.forward(**ca_state)
    # ra_state = reasoning_agent.forward(**ca_state)



    # from langgraph_swarm import create_swarm, SwarmState
    # from langchain.agents import create_agent

    # create_swarm(
    #     [
    #         ConversationAgent(llm),
    #         DomainAgent(llm),
    #     ],
    #     default_active_agent = "ConversationAgent"
    # )

    # prompts = util.loadInitialPrompts("./prompts")

    # mlflow.genai.set_prompt_model_config(
    #     name="my-prompt",
    #     version=1,
    #     model_config={"model_name": "gpt-4", "temperature": 0.7},
    # )

    # VECTOR_ENDPOINT = "https://example.com/vectorize"   # <-- change
    # UPLOAD_DISABLED = False
    # # HEADERS = {"Authorization": "Bearer YOUR_TOKEN"}

    # TODO
    # @scorer
    # def eval_question_extraction(user_question, generated_question):

    # from agent_system.prompts import load_all_prompts
    # load_all_prompts()
    # print("mlflow - prompts loaded")

    # mlflangchain.log_model(LLM)

    # LLM.invoke([{"role": "user", "content": "give me a list of ingreidence for a general curry dish"}])

    # genai.create_dataset()
    