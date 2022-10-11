import os

import networkx as nx
import numpy as np
from types import SimpleNamespace

from core.backend.base import ElemwiseOps, ProcessingOps, ReduceOps, ViewOps, CreationOps
from env import DEBUG, OPT_CONSTANT_FOLDING
from utils.helper import varnamegetter

class GraphOptimizer:
    def __init__(self, root):
        assert root.is_lazy
        self.root = root
        self._constant_folding_visit_flag = {}

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

    def _constant_folding(self, node):
        # TODO: refactor
        from core.backend.numpy import NPArray
        from core.backend.opencl import CLArray, cl
        if node.constant_flag and node.constant_value is not None:
            if DEBUG: print(f"[DEBUG] {id(node)} return True")
            return True

        dep_constant_flags = {}
        for name, dep_node in node.op_info.operands.items():
            if id(dep_node) not in self._constant_folding_visit_flag:
                flag = self._constant_folding(dep_node)
                self._constant_folding_visit_flag[id(dep_node)] = flag
            dep_constant_flags[name] = self._constant_folding_visit_flag[id(dep_node)]

        if any(dep_constant_flags.values()):
            if isinstance(node.op_info.operator, ViewOps):
                if DEBUG: print(f"[DEBUG] view node update id {id(node)}")
                assert len(dep_constant_flags) == 1
                dep_node = list(node.op_info.operands.values())[0]
                node.op_info = SimpleNamespace(operator=None, operands={})
                node.is_lazy = False
                node.constant_flag = True
                constant_value = dep_node.constant_value
                node.constant_value = constant_value
                data = np.asarray(constant_value, dtype=node.dtype)
                if isinstance(node, NPArray):
                    node.data = data
                elif isinstance(node, CLArray):
                    node.buffer = cl.alloc_buffer((), node.dtype, data)

            if isinstance(node.op_info.operator, ElemwiseOps):
                if DEBUG: print(f"[DEBUG {id(node)} elemwise node update, original operands: {node.op_info.operands}")
                code = node.op_info.code
                for name, dep_node in node.op_info.operands.items():
                    if not dep_constant_flags[name]: continue
                    #print(code)
                    code = code.replace(name, f"{dep_node.constant_value}f")
                #print(code)
                node.op_info.code = code
                for name in dep_constant_flags:
                    if not dep_constant_flags[name]: continue
                    del node.op_info.operands[name]

                if all(dep_constant_flags.values()) and len(dep_constant_flags) > 1:
                    if DEBUG: print(f"[DEBUG] elemwise node fully update id {id(node)}")
                    node.op_info = SimpleNamespace(operator=None, operands={})
                    node.is_lazy = False
                    node.constant_flag = True
                    constant_value = eval(code.replace("f", "").replace("exp", "np.exp"))
                    node.constant_value = constant_value
                    data = np.asarray(constant_value, dtype=node.dtype)
                    if isinstance(node, NPArray):
                        node.data = data
                    elif isinstance(node, CLArray):
                        node.buffer = cl.alloc_buffer((), node.dtype, data)

            if isinstance(node.op_info.operator, ProcessingOps):
                # NODE: never happen
                pass

        constant_flag = len(dep_constant_flags) > 1 and all(dep_constant_flags.values())
        if DEBUG: print(f"[DEBUG] {id(node)} return {constant_flag} dep_constant_flags: {dep_constant_flags} final operands: {node.op_info.operands}")
        return constant_flag

    def visualize(self, suffix=""):
        color_map = {ReduceOps: "#ecc30b", ElemwiseOps: "#84bcda", ProcessingOps: "#f37748", ViewOps: "#e5e5e5"}
        def build_graph(node, G):
            if node is None: return G
            nid = id(node)
            if nid in G.nodes: return G
            G.add_node(nid)
            label = f"{node.shape}\n{nid}\nC:{node.c_contiguous} F:{node.f_contiguous}"
            if OPT_CONSTANT_FOLDING:
                if node.constant_value is not None:
                    label += f"\nCONSTANT={node.constant_value}"
            if node.op_info.operator is not None: label += f"\n{node.op_info.operator.name}"
            G.nodes[nid]["label"] = label
            G.nodes[nid]["shape"] = "box"
            G.nodes[nid]["style"] = "filled, dashed" if node.is_lazy else "filled"
            G.nodes[nid]["fillcolor"] = color_map.get(type(node.op_info.operator), "#ffffff")
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
