from logger import housing_logger
import time


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
