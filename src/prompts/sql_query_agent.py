from .base import LLMPromptTemplate

SQL_SYSTEM_MESSAGE = """
You are an expert SQL assistant for Hong Kong housing data.
- Generate SINGLE valid SQL queries only (e.g., SELECT, no DDL).
- Do not explain, just provide the SQL.
- Use table and column names exactly as given.
- Return query in plain text, no markdown or code blocks.
"""

SQL_USER_MESSAGE_TEMPLATE = """
Given the following user question, generate a SQL query to answer it.
Use the table schema information provided.
User Question: {user_question}
Table name: {table_name}
Table Info: {table_info}
Table Schema: {db_schema}
"""

# TODO: Make this dynamic based on the actual table being queried
# For now, it's hardcoded for the "estate_info" table

ESTATE_INFO_TABLE_INFO = """
This table contains information about various estates in Hong Kong.
"""

def create_sql_prompt(user_question: str, db_schema: str) -> LLMPromptTemplate:
    user_message = SQL_USER_MESSAGE_TEMPLATE.format(
        user_question=user_question,
        table_name="estate_info",
        table_info=ESTATE_INFO_TABLE_INFO,
        db_schema=db_schema # TODO: precache db schema
    )
    return LLMPromptTemplate(
        user_messages=user_message,
        system_messages=SQL_SYSTEM_MESSAGE
    )