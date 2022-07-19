import os
import string
import networkx as nx
from env import OPT1, DEBUG
from utils.helper import varnamegetter
from core.backend.base import ElemwiseOps, ProcessingOps, ReduceOps, ViewOps, CreationOps

class GraphOptimizer:
    def __init__(self, target_node):
        assert target_node.is_lazy
        self.target_node = target_node
        varnamegetter.reset()

    def build(self, node=None):
        if node is None: node = self.target_node
        # reset outdegree
        for name, dep_node in node.op_info.operands.items():
            dep_node.outdegree = 0
            dep_node.is_visited = False
            self.build(dep_node)
        for name, dep_node in node.op_info.operands.items():
            dep_node.outdegree += 1
            self.build(dep_node)
        return self

    def _merge_elemwise(self, node):
        """element-wise ops (unary or binary) can be merged, thus reduce kernel calls. Consider the following computational graph.
        `a = b + c; d = a * e; ret = exp(d)`
        It has three element-wise kernel ops, i.e. add, mul and exp. We can merge that into a single kernel call, i.e. exp((b+c)*e).
        """
        operands = {}
        operator = node.op_info.operator
        for name, dep_node in node.op_info.operands.items():
            if not dep_node.is_lazy:
                new_name = varnamegetter.get()
                operands[new_name] = dep_node
                if type(operator) is ElemwiseOps:
                    node.op_info.code = node.op_info.code.replace(name, new_name)
            else:
                if not dep_node.is_visited: self._merge_elemwise(dep_node)
                if type(operator) is ElemwiseOps and type(dep_node.op_info.operator) is ElemwiseOps and \
                        dep_node.outdegree == 1:
                    operands.update(dep_node.op_info.operands)
                    experssion = f"({dep_node.op_info.code})"
                    node.op_info.code = node.op_info.code.replace(name, experssion)
                else:
                    new_name = varnamegetter.get()
                    operands[new_name] = dep_node
                    if type(operator) is ElemwiseOps:
                        node.op_info.code = node.op_info.code.replace(name, new_name)
        node.is_visited = True
        node.op_info.operands = operands

    def _simplify_arithmetic(self):
        pass

    def _operation_fusion(self):
        pass

    def optimize(self):
        if OPT1: self._merge_elemwise(node=self.target_node)

    def visualize(self, prefix=""):
        color_map = {ReduceOps: "#ecc30b", ElemwiseOps: "#84bcda", ProcessingOps: "#f37748"}
        def build_nx_graph(node, G):
            if node is None: return G
            nid = id(node)
            if nid in G.nodes: return G
            G.add_node(nid)
            G.nodes[nid]["label"] = f"{node.shape}\n{nid}"
            if hasattr(node.op_info, "code"):
                G.nodes[nid]["label"] += f"\n{node.op_info.code}"
            lazy = True
            if node.is_lazy:
                G.nodes[nid]["fillcolor"] = color_map[type(node.op_info.operator)]
            else:
                G.nodes[nid]["fillcolor"] = "#ffffff"; lazy = False
            G.nodes[nid]["style"] = "filled, dashed" if not lazy else "filled"
            for name, subnode in node.op_info.operands.items():
                G = build_nx_graph(subnode, G)
                edge = (id(subnode), nid)
                if edge not in G.edges:
                    G.add_edge(*edge, cnt=1, label=name)
            return G
        G = nx.DiGraph()
        G = build_nx_graph(self.target_node, G)
        mode = prefix + "_array"
        nx.drawing.nx_pydot.write_dot(G, f"/tmp/{mode}.dot")
        os.system(f"dot -Tsvg /tmp/{mode}.dot -o /tmp/{mode}.svg")
        print(f"[GRAPH] save to /tmp/{mode}.svg")

