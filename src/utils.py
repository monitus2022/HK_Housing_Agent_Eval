import duckdb
from logger import housing_logger
from config import settings
import time


def connect_duckdb(function: callable) -> callable:
    """
    Wrapper function to connect to DuckDB, and close the connection after the function call.
    """

    def wrapper(*args, **kwargs):
        db_path = settings.duckdb_path
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


def timer(function: callable) -> callable:
    """
    Wrapper function to time the execution of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        housing_logger.info(
            f"Function '{function.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result

    return wrapper
