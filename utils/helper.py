import time

def timer(func):
    def wrapper(*args, **kwargs):
        ts = time.time()
        ret = func(*args, **kwargs)
        cost = time.time() - ts
        return ret, cost
    return wrapper

def genname(prefix, *args):
    return f"{prefix}_" + "_".join(str(id(ts))[-4:] for ts in args)