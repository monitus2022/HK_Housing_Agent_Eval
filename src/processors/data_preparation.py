from utils import connect_duckdb
from logger import housing_logger
from .sql_queries import *


@connect_duckdb
def train_test_split_data(
    conn, table_name: str, 
    test_size: float = 0.2, random_state: int = 42
) -> None:
    """
    Split the DuckDB table into train and test sets,
    and store them back in DuckDB.
    """
    # Get the total row count
    total_rows = get_total_rows_from_table(conn, table_name)
    test_count = int(total_rows * test_size)
    train_count = total_rows - test_count

    # Split the data and create new tables
    create_train_test_split_tables(
        conn, table_name, test_size, random_state
    )

    housing_logger.info(
        f"Data split into train ({train_count}) and test ({test_count}) sets."
    )
