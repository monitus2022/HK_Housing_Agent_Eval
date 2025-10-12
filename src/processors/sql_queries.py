from logger import housing_logger
from duckdb import DuckDBPyConnection


def get_total_rows_from_table(conn: DuckDBPyConnection, table_name: str) -> int:
    """
    Get the total number of rows in a DuckDB table.

    Args:
        conn: DuckDB connection object.
        table_name (str): Name of the table.

    Returns:
        int: Total number of rows in the table.
    """
    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return result[0] if result else 0


def drop_table_if_exists(conn: DuckDBPyConnection, table_name: str) -> None:
    """
    Drop a table if it exists in DuckDB.

    Args:
        conn: DuckDB connection object.
        table_name (str): Name of the table to drop.
    """
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    housing_logger.info(f"Table already exists, dropped table: {table_name}")


def create_train_test_split_tables(
    conn: DuckDBPyConnection,
    table_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Split a table into exact, complementary train and test sets using SAMPLE.
    Test is a random sample; train is the exact complement.
    """
    test_percentage = test_size * 100

    # Drop existing tables
    drop_table_if_exists(conn, f"{table_name}_test_data")
    drop_table_if_exists(conn, f"{table_name}_train_data")

    # Create test set (random sample with seed for reproducibility)
    test_query = f"""
        CREATE TABLE {table_name}_test_data AS
        SELECT * FROM {table_name} USING SAMPLE {test_percentage}% (SYSTEM, {random_state})
    """
    conn.execute(test_query)

    # Create train set (exact complement using EXCEPT)
    train_query = f"""
        CREATE TABLE {table_name}_train_data AS
        SELECT * FROM {table_name}
        EXCEPT
        SELECT * FROM {table_name}_test_data
    """
    conn.execute(train_query)

    housing_logger.info(
        f"Data split into complementary train and test sets with test size ~{test_size}."
    )
