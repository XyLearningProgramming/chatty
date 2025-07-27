import functools


def singleton(func):
    """
    Decorator for a factory function.
    Caches the first return value in func._instance
    and always returns that thereafter.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(func, "_instance"):
            # First call: create & stash
            func._instance = func(*args, **kwargs)
        return func._instance

    return wrapper
