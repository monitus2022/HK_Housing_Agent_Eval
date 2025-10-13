from logger import housing_logger
from duckdb import DuckDBPyConnection

class QueryExecutor:
    def __init__(self, conn: DuckDBPyConnection):
        self.conn = conn
    
    def execute_query(self, query: str) -> any:
        try:
            housing_logger.info(f"Executing query: {query}")
            return self.conn.execute(query).fetchall()
        except Exception as e:
            housing_logger.error(f"Error executing query: {e}")
            raise

    def get_schema_from_table(self, table_name: str) -> str:
        """
        get the schema of a table in DuckDB and return as a string for llm prompt
        """
        result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        if not result:
            housing_logger.error(f"Table '{table_name}' does not exist or has no schema.")
            raise ValueError(f"Table '{table_name}' does not exist or has no schema.")
        
        schema_lines = [f"{col[0]}: {col[1]}" for col in result]
        schema_str = "\n".join(schema_lines)
        housing_logger.info(f"Obtained schema for table '{table_name}' with {len(result)} columns.")
        return schema_str

    def get_total_rows_from_table(self, table_name: str) -> int:
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0

    def drop_table_if_exists(self, table_name: str) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        housing_logger.info(f"Table already exists, dropped table: {table_name}")

    def create_train_test_split_tables(self, table_name: str, test_size: float = 0.2, random_state: int = 42) -> None:
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")
        
        test_percentage = test_size * 100
        
        # Drop existing tables
        self.drop_table_if_exists(f"{table_name}_train_data")
        self.drop_table_if_exists(f"{table_name}_test_data")
        
        # Create test set with random sample
        self.conn.execute(f"""
        CREATE TABLE {table_name}_test_data AS
        SELECT * FROM {table_name} USING SAMPLE {test_percentage}% (SYSTEM, {random_state})
        """)
        
        # Create train set as the complement (remaining rows)
        self.conn.execute(f"""
        CREATE TABLE {table_name}_train_data AS
        SELECT * FROM {table_name}
        WHERE rowid NOT IN (SELECT rowid FROM {table_name}_test_data)
        """)
        
        train_rows = self.get_total_rows_from_table(f"{table_name}_train_data")
        test_rows = self.get_total_rows_from_table(f"{table_name}_test_data")
        housing_logger.info(f"Data split: Train set ({train_rows} rows), Test set ({test_rows} rows), Test size ~{test_size}.")