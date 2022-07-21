import itertools
import os
import string
import time
from collections import defaultdict

import networkx as nx

def timer(func):
    def wrapper(*args, **kwargs):
        ts = time.monotonic()
        ret = func(*args, **kwargs)
        cost = time.monotonic() - ts
        return ret, cost
    return wrapper

def genname(prefix, *args):
    return f"{prefix}_" + "_".join(str(id(ts))[-4:] for ts in args)

class VarNameGetter:
    def __init__(self):
        letters = tuple(string.ascii_lowercase)
        candidates = itertools.product(*([letters]*3))  # 26^4
        self.candidates = tuple("".join(s) for s in candidates)
        self.reset()

    def get(self):
        name = self.candidates[self.idx]
        self.idx += 1
        return name

    def reset(self):
        self.idx = 0

varnamegetter = VarNameGetter()

class KernelStat:
    def __init__(self):
        self.reset()
    def reset(self):
        self._counter = defaultdict(lambda : defaultdict(int))
    def log(self, operator):
        kerneltype = type(operator)
        self._counter[kerneltype][operator.name] += 1
    def get(self, kernel_type):
        return self._counter[kernel_type]
    def total(self):
        return sum(sum(v.values()) for k, v in self._counter.items())
    @property
    def info(self):
        info = {}
        for k, v in self._counter.items():
            info[k] = dict(v)
        return info

kernelstat = KernelStat()

