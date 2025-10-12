import duckdb
from logger import housing_logger
from config import housing_settings

def connect_duckdb(function: callable) -> callable:
    """
    Wrapper function to connect to DuckDB, and close the connection after the function call.
    """
    def wrapper(*args, **kwargs):
        db_path = housing_settings.DUCKDB_PATH
        conn = duckdb.connect(database=db_path)
        try:
            housing_logger.info(f"Connected to DuckDB at {db_path}")
            return function(conn, *args, **kwargs)
        except Exception as e:
            housing_logger.error(f"Error occurred: {e}")
            raise
        finally:
            conn.close()
    return wrapper
