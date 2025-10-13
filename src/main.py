from agents.langchain_sql_agents import LangChainSqlAgent
from llm.base import LLMPromptTemplate

def main():
    agent = LangChainSqlAgent()
    agent.set_model(model_name="openai")
    agent.setup_agent(model_params=None)

    prompt = LLMPromptTemplate(user_messages="Which are the 3 northernmost estates?")
    response = agent.act(prompt)
    print(response)

if __name__ == "__main__":
    main()