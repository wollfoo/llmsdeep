import logging
from functools import wraps
def trace_power_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = kwargs.get("logger", logging.getLogger(__name__))

        power_limit = None
        # Kiểm tra nếu power_limit được truyền qua kwargs
        if "power_limit" in kwargs:
            power_limit = kwargs["power_limit"]
            logger.debug(f"[TRACE] power_limit được truyền qua kwargs: {power_limit}")
        else:
            # Kiểm tra nếu power_limit được truyền qua args
            func_args = func.__code__.co_varnames
            if "power_limit" in func_args:
                index = func_args.index("power_limit")
                if index < len(args):
                    power_limit = args[index]
                    logger.debug(f"[TRACE] power_limit được truyền qua args: {power_limit}")

        # Log cảnh báo nếu không tìm thấy power_limit
        if power_limit is None:
            logger.warning("Không tìm thấy giá trị power_limit được truyền.")
            # Có thể ném lỗi hoặc trả về một giá trị mặc định
            raise ValueError("Giá trị power_limit là bắt buộc!")

        # Gọi hàm gốc
        return func(*args, **kwargs)
    return wrapper
