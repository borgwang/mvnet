import copy
import os
from collections import defaultdict

import networkx as nx
import numpy as np
from types import SimpleNamespace

from core.backend.base import ElemwiseOps, ProcessingOps, ReduceOps, ViewOps, CreationOps
from env import *
from utils.helper import varnamegetter

class GraphOptimizer:
    def __init__(self, root):
        assert root.is_lazy
        self.root = root
        varnamegetter.reset()

    def _elemwise_fusion(self, node):

        def elemwise_fusion(node):
            for name in list(node.op_info.operands):
                dep_node = node.op_info.operands[name]
                if not dep_node.is_lazy:
                    continue
                if not visit_flag[id(dep_node)]:
                    elemwise_fusion(dep_node)
                if type(node.op_info.operator) is ElemwiseOps and type(dep_node.op_info.operator) is ElemwiseOps and outdegree[id(dep_node)] == 1:
                    node.op_info.operands.pop(name)
                    node.op_info.operands.update(dep_node.op_info.operands)
                    node.op_info.code = node.op_info.code.replace(name, f"({dep_node.op_info.code})")
            visit_flag[id(node)] = True

        def update_outdegree(node):
            if visit_flag[id(node)]: return
            for name, dep_node in node.op_info.operands.items():
                outdegree[id(dep_node)] += 1
                update_outdegree(dep_node)
            visit_flag[id(node)] = True

        outdegree = defaultdict(int)
        visit_flag = defaultdict(bool)
        update_outdegree(node)
        visit_flag = defaultdict(bool)
        elemwise_fusion(node)

    def _rename_operands(self, node):
        def rename_operands(node):
            newoperands = {}
            for name, dep_node in node.op_info.operands.items():
                if not visit_flag[id(dep_node)]:
                    rename_operands(dep_node)
                new_name = name_dict[id(dep_node)]
                newoperands[new_name] = dep_node
                if type(node.op_info.operator) is ElemwiseOps:
                    node.op_info.code = node.op_info.code.replace(name, new_name)
            node.op_info.operands = newoperands
            visit_flag[id(node)] = True

        visit_flag = defaultdict(bool)
        name_dict = defaultdict(varnamegetter.get)
        rename_operands(node)

    def _constant_folding(self, node):
        def constant_folding(node):
            if node.constant_value is not None:
                return True
            dep_const_flags = {}
            for name, dep_node in node.op_info.operands.items():
                if id(dep_node) not in cache:
                    flag = constant_folding(dep_node)
                    cache[id(dep_node)] = flag
                dep_const_flags[name] = cache[id(dep_node)]

            const_flag = False
            if any(dep_const_flags.values()):
                if isinstance(node.op_info.operator, ViewOps):
                    dep_node = next(iter(node.op_info.operands.values()))
                    node.to_constant(dep_node.constant_value)
                    const_flag = True

                if isinstance(node.op_info.operator, ElemwiseOps):
                    if all(dep_const_flags.values()) and len(dep_const_flags) > 1:  # NOTE: skip unary ops
                        expr = node.op_info.code
                        for name, dep_node in node.op_info.operands.items():
                            expr = expr.replace(name, f"{dep_node.constant_value:.15f}f")
                        node.to_constant(eval(expr.replace("f", "")))
                        const_flag = True
            return const_flag

        cache = {}
        constant_folding(node)

    def visualize(self, graph_name):
        colors = {ReduceOps: "#ecc30b", ElemwiseOps: "#84bcda", ProcessingOps: "#f37748", ViewOps: "#e5e5e5"}
        def build_graph(node, G):
            if node is None: return G
            if id(node) in G.nodes: return G
            G.add_node(id(node))
            label = (f"{node.shape}\n"
                     f"{id(node)}\n"
                     f"C:{int(node.c_contiguous)} F:{int(node.f_contiguous)}")
            if node.constant_value is not None:
                label += f"\nCONSTANT={node.constant_value}"
            if node.op_info.operator is not None: label += f"\n{node.op_info.operator.name}"
            G.nodes[id(node)]["label"] = label
            G.nodes[id(node)]["shape"] = "box"
            G.nodes[id(node)]["style"] = "filled, dashed" if node.is_lazy else "filled"
            G.nodes[id(node)]["fillcolor"] = colors.get(type(node.op_info.operator), "#ffffff")
            for name, subnode in node.op_info.operands.items():
                G = build_graph(subnode, G)
                edge = (id(subnode), id(node))
                if edge not in G.edges:
                    G.add_edge(*edge, cnt=1, label=name)
            return G
        G = nx.DiGraph()
        G = build_graph(self.root, G)
        nx.drawing.nx_pydot.write_dot(G, f"/tmp/{graph_name}.dot")
        os.system(f"dot -Tsvg /tmp/{graph_name}.dot -o /tmp/{graph_name}.svg")
        print(f"[GRAPH] save to /tmp/{graph_name}.svg")
