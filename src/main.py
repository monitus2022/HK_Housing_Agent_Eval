from agents.sql_query_agent import SqlQueryAgent
from prompts import create_sql_prompt
from pprint import pprint

def main():
    agent = SqlQueryAgent()
    table_schema = agent.query_executor.get_schema_from_table("estate_info")
    agent.set_model(model_name="llama_small_free")
    agent.setup_agent(model_params=None)

    prompt = create_sql_prompt(
        user_question="What are the 3 northernmost estates in Hong Kong?",
        db_schema=table_schema
    )
    response = agent.act(prompt)
    pprint(response.content)
    return response

if __name__ == "__main__":
    main()