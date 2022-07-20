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
        candidates = itertools.product(*([letters]*2))  # 26^4
        self.candidates = tuple("".join(s) for s in candidates)
        self.reset()

    def get(self):
        name = self.candidates[self.idx]
        self.idx += 1
        return name

    def reset(self):
        self.idx = 0

varnamegetter = VarNameGetter()

# tensor graph (high level)
def get_tensor_graph(start):
    def build_graph(node, G):
        if id(node) in G.nodes:
            return G
        G.add_node(id(node))
        G.nodes[id(node)]["label"] = f"{node.shape}\n{id(node.array)}"
        for dep in node.dependency:
            subnode = dep["tensor"]
            G = build_graph(subnode, G)
            edge = (id(subnode), id(node))
            if edge in G.edges:
                cnt = nx.get_edge_attributes(G, "cnt")[edge]["cnt"]
                nx.set_edge_attributes(G, {edge: {"cnt": cnt+1}})
            else:
                opname = node.name.split("_")[0]
                G.add_edge(*edge, cnt=1, label=opname)
        return G
    G = nx.DiGraph()
    G = build_graph(start, G)
    mode = "tensor"
    nx.drawing.nx_pydot.write_dot(G, f"/tmp/{mode}.dot")
    os.system(f"dot -Tsvg /tmp/{mode}.dot -o /tmp/{mode}.svg")
    print(f"[GRAPH] save to /tmp/{mode}.svg")

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

