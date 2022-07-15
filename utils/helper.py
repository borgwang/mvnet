import time

def timer(func):
    def wrapper(*args, **kwargs):
        ts = time.monotonic()
        ret = func(*args, **kwargs)
        cost = time.monotonic() - ts
        return ret, cost
    return wrapper

def genname(prefix, *args):
    return f"{prefix}_" + "_".join(str(id(ts))[-4:] for ts in args)


import string
class VarNameGetter:
    def __init__(self):
        self.candidate = list(string.ascii_lowercase)
        self.reset()

    def get(self, obj):
        if obj in self.cache:
            return self.cache[obj]
        name = self.candidate[self.idx]
        self.cache[obj] = name
        self.idx += 1
        return name

    def reset(self):
        self.cache = {}
        self.idx = 0

varnamegetter = VarNameGetter()
