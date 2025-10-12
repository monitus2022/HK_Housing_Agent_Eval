from utils import connect_duckdb
from logger import housing_logger


@connect_duckdb
def train_test_split_data(
    conn, table_name: str, test_size: float = 0.2, random_state: int = 42
) -> None:
    """
    Import table from Duckdb, split into train and test sets, 
    and store them back in DuckDB.
    """
    # Load data from DuckDB table
    df = conn.execute(
        f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}"
    ).fetchdf()

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate the number of test samples
    test_count = int(len(df) * test_size)

    # Split the DataFrame into train and test sets
    test_df = df.iloc[:test_count]
    train_df = df.iloc[test_count:]

    # Store the train and test sets back in DuckDB
    conn.execute(f"DROP TABLE IF EXISTS {table_name}_train_data")
    conn.execute(f"DROP TABLE IF EXISTS {table_name}_test_data")
    conn.register(f"{table_name}_train_data", train_df)
    conn.register(f"{table_name}_test_data", test_df)

    housing_logger.info(
        f"""Data split into train ({len(train_df)}) 
        and test ({len(test_df)}) sets and stored in DuckDB."""
    )
