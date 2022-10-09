import os
import networkx as nx
from env import DEBUG
from utils.helper import varnamegetter
from core.backend.base import ElemwiseOps, ProcessingOps, ReduceOps, ViewOps, CreationOps

class GraphOptimizer:
    def __init__(self, root):
        assert root.is_lazy
        self.root = root
        varnamegetter.reset()

    def build(self, node=None):
        def _reset_visit(node):
            for name, dep_node in node.op_info.operands.items():
                dep_node.is_visited = False
                _reset_visit(dep_node)
        def _reset_outdegree(node):
            for name, dep_node in node.op_info.operands.items():
                dep_node.outdegree = 0
                _reset_outdegree(dep_node)
        def _build(node):
            if node.is_visited: return
            for name, dep_node in node.op_info.operands.items():
                dep_node.outdegree += 1
                _build(dep_node)
                dep_node.is_visited = True

        if node is None: node = self.root
        _reset_outdegree(node)
        _build(node)
        _reset_visit(node)

    def _remove_contiguous(self, node):
        operator = node.op_info.operator
        if operator == ElemwiseOps.NOOP:
            assert len(node.op_info.operands.values()) == 1, "ElemwiseOps.NOOP should have only one input"

    def _merge_elemwise(self, node):
        """element-wise ops (unary or binary) can be merged, thus reduce kernel calls. Consider the following computational graph.
        `a = b + c; d = a * e; ret = exp(d)`
        Three element-wise kernel ops can be merged into one single kernel call, i.e. ret = exp((b + c) * e).
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
                if type(operator) is ElemwiseOps and type(dep_node.op_info.operator) is ElemwiseOps and dep_node.outdegree == 1:
                    operands.update(dep_node.op_info.operands)
                    experssion = f"({dep_node.op_info.code})"
                    node.op_info.code = node.op_info.code.replace(name, experssion)
                    if DEBUG: print(f"DEBUG replace expression {id(node)} {name} -> {experssion}")
                else:
                    new_name = varnamegetter.get()
                    operands[new_name] = dep_node
                    if type(operator) is ElemwiseOps:
                        node.op_info.code = node.op_info.code.replace(name, new_name)
                        if DEBUG: print(f"DEBUG replace name {id(node)} {name} -> {new_name}")
        node.is_visited = True
        node.op_info.operands = operands

    def _simplify_arithmetic(self):
        pass

    def _operation_fusion(self):
        pass

    def optimize(self):
        pass

    def visualize(self, suffix=""):
        color_map = {ReduceOps: "#ecc30b", ElemwiseOps: "#84bcda", ProcessingOps: "#f37748", ViewOps: "#e5e5e5"}
        def build_graph(node, G):
            if node is None: return G
            nid = id(node)
            if nid in G.nodes: return G
            G.add_node(nid)
            label = f"{node.shape}\n{nid}"
            if node.op_info.operator is not None: label += f"\n{node.op_info.operator.name}"
            #if hasattr(node.op_info, "code"): label += f"\n{node.op_info.code}"
            G.nodes[nid]["label"] = label
            G.nodes[nid]["shape"] = "box"
            G.nodes[nid]["style"] = "filled, dashed" if not node.is_lazy else "filled"
            G.nodes[nid]["fillcolor"] = color_map[type(node.op_info.operator)] if node.is_lazy else "#ffffff"
            for name, subnode in node.op_info.operands.items():
                G = build_graph(subnode, G)
                edge = (id(subnode), nid)
                if edge not in G.edges:
                    G.add_edge(*edge, cnt=1, label=name)
            return G
        G = nx.DiGraph()
        G = build_graph(self.root, G)
        name = "net"
        if suffix: name += "_" + suffix
        nx.drawing.nx_pydot.write_dot(G, f"/tmp/{name}.dot")
        os.system(f"dot -Tsvg /tmp/{name}.dot -o /tmp/{name}.svg")
        print(f"[GRAPH] save to /tmp/{name}.svg")
