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




if __name__ == "__main__":

    mlfopenai.autolog()
    # mlflangchain.autolog()

    dotenv.load_dotenv()

    init_mlflow(experiment_name="test-experiment")

    from agent_system.util import OpenAI_LLM
    llm = OpenAI_LLM(os.environ["OPENAI_ENDPOINT"])


    conversation_agent = agents.Conversation_Agent(llm=llm, agents=[
        # agents.CAG_Agent(llm),
        # agents.RAG_Agent(llm),
        # agents.Reasoning_Agent(llm)
    ])

    ca_state = agents.CA_State(**
        {
            "CurrentConversation": [HumanMessage("")]
        }
    )

    agent_states = {k: None for k, a in conversation_agent.agents.items()}


    for _ in range(10):

        ca_state = agents.CA_State(**ca_state.model_dump())
        ca_state = agents.CA_State(**conversation_agent.forward(**ca_state.model_dump()))

        print(f"answer     => {ca_state.CurrentConversation[-1].content}")


        # print(f"next_agent => {ca_state.HandofAgent}")

        # agent_name = ca_state.HandofAgent.gewählter_agent.name

        # # if ca_state.HandofAgent.verwende_agent:

        # next_agent = conversation_agent.agents[agent_name]
        # current_agent_state = agent_states[agent_name]
        # current_agent_state.CurrentConversation.append(ca_state.CurrentConversation[:-2])


        # print(current_agent_state.model_dump())

        # new_agent_state = next_agent.forward(**current_agent_state)
        # agent_states[ca_state.HandofAgent.name] = new_agent_state

        # print(f"Agent out  => {next_agent}")




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
    