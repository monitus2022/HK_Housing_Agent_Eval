from agents.sql_query_agent import SqlQueryAgent
from llm import LLMPromptTemplate
from config import settings

def main():
    agent = SqlQueryAgent()
    agent.set_model(model_name="llama_small_free")
    agent.setup_db(db_path=settings.duckdb_path, db_type="duckdb")
    prompt = LLMPromptTemplate(
        user_messages="Find the top 5 estates by transaction volume.",
        system_prompt="You must use tools to answer. Do not say 'I don't know.' Always call a tool."
    )
    agent.setup_agent(model_params=None)
    response = agent.act(prompt)
    print("Agent Response:", response)

if __name__ == "__main__":
    main()