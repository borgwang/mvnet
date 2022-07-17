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

# array graph (low level)
def get_array_graph(start):
    color_map = {"reduce": "#ecc30b", "elementwise": "#84bcda", "matmul": "#f37748",
                 "contiguous": "#d56062"}
    def build_graph(node, G):
        if node is None:
            return G
        nid = id(node)
        operator = node.operator
        if nid in G.nodes: return G
        G.add_node(nid)
        G.nodes[nid]["label"] = f"{node.shape}\n{node.outdegree}\n{nid}\n"
        dashed = False
        if not operator:
            G.nodes[nid]["fillcolor"] = "#ffffff"; dashed = True
        else:
            G.nodes[nid]["fillcolor"] = color_map[operator["type"]]
        G.nodes[nid]["style"] = "filled, dashed" if dashed else "filled"
        for name, subnode in node.operands.items():
            G = build_graph(subnode, G)
            edge = (id(subnode), nid)
            if edge not in G.edges:
                opname = operator["type"][:3].upper()
                if "code" in operator: opname += ("_" + operator["code"])
                G.add_edge(*edge, cnt=1, label=opname)
        return G
    G = nx.DiGraph()
    G = build_graph(start, G)
    mode = "array"
    nx.drawing.nx_pydot.write_dot(G, f"/tmp/{mode}.dot")
    os.system(f"dot -Tsvg /tmp/{mode}.dot -o /tmp/{mode}.svg")
    print(f"[GRAPH] save to /tmp/{mode}.svg")


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

class GraphOptimizer:
    def __init__(self):
        pass

    def build(self, node):
        operator = node.operator
        for name, subnode in node.operands.items():

            pass

    def __combine_elementwise(self):
        pass

    def __simplify_arithmetic(self):
        pass

    def __operation_fusion(self):
        pass

    def optimize(self):
        pass

    def visualize(self):
        pass


graphoptimizer = GraphOptimizer()
