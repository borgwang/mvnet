import time
import os
import networkx as nx
import string

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

def plot_graph(start):
    def build_graph(node, G):
        if id(node) in G.nodes:
            return G
        G.add_node(id(node))
        G.nodes[id(node)]["label"] = f"{node.shape}\n{node.array.outdegree}\n{id(node.array)}"
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
    nx.drawing.nx_pydot.write_dot(G, '/tmp/net.dot')
    os.system('dot -Tsvg /tmp/net.dot -o /tmp/net.svg')
    print("save to /tmp/net.svg")


from collections import defaultdict

class KernelStat:
    def __init__(self):
        self.reset()
    def reset(self):
        self.counter = defaultdict(int)
    def log(self, name):
        self.counter[name] += 1
    def get(self, name):
        return self.counter[name]
    def total(self):
        return sum(self.counter.values())

kernelstat = KernelStat()

