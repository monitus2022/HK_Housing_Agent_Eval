import duckdb
from logger import housing_logger
from config import settings

class DuckDBManager:
    def __init__(self):
        self.db_path = settings.duckdb_path
        self.conn = duckdb.connect(database=self.db_path)
        if not self.conn:
            housing_logger.error(f"Failed to connect to DuckDB at {self.db_path}")
            raise ConnectionError(f"Failed to connect to DuckDB at {self.db_path}")
        housing_logger.info(f"Connected to DuckDB at {self.db_path}")

    def close_connection(self):
        if self.conn:
            housing_logger.info(f"Closing DuckDB connection at {self.db_path}")
            self.conn.close()
            self.conn = None