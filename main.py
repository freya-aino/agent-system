import dotenv
import os
import mlflow
import mlflow.openai as mlfopenai
import mlflow.langchain as mlflangchain

from langchain.messages import HumanMessage

from agent_system.agents import *
from agent_system.util import init_mlflow


# TODO - add prompt version
# TODO - add production prompt # prompt = mlflow.genai.load_prompt("prompts:/my_prompt@production")
# TODO - generalize the tool calls to using tool calling strategy so that models that dont have native structured output can also use tools
# class ProductReview(BaseModel):
#     """Analysis of a product review."""
#     rating: int | None = Field(description="The rating of the product", ge=1, le=5)
#     sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
#     key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")



if __name__ == "__main__":

    dotenv.load_dotenv()

    # init_mlflow()

    from agent_system.util import OpenAI_LLM
    llm = OpenAI_LLM(base_url="gpt-4.1")
    
    # AgentAgent(
    #     [
    #         ConversationAgent(llm),
    #         DomainAgent(llm)
    #     ]
    # )

    ca = ConversationAgent(llm)
    da = DomainAgent(llm)
    ka = KnowledgeAgent(llm)
    ra = ReasoningAgent(llm)
    va = ValidationAgent(llm)

    print("---\n".join([f"# {k}\n{k}\n" for k in ["a", "b"]]))

    


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


    # conversationAgent = agents.ConversationAgent(LLM)
    # knowledgeAgent = agents.KnowledgeAgent(LLM)
    
    # for _ in range(10):
    #     in_ = input("ENTER PROMPT: ")
        
    #     ca_out = conversationAgent(in_)

    #     print(f"Classification:         {ca_out.Classification}")
    #     print(f"Information Key Points: {ca_out.InformationKeyPoints}")

    #     ka_out = knowledgeAgent(ca_out)

    #     print(f"System Search Queries:  {ka_out.SystemSearchQueries}")
    #     print(f"Document Chunks:        {ka_out.DocumentChunksInContext}")

    #     print("---")
    #     print(ca_out.Conversation[-1].content)
    #     print("---")

    # print(out)
    # print(conversationAgent.state)


    # genai.create_dataset()
    