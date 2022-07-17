import os
import string
import networkx as nx
from env import OPT1
from utils.helper import varnamegetter

class GraphOptimizer:
    def __init__(self, target_node):
        self.target_node = target_node
        varnamegetter.reset()

    def build(self, node=None):
        # recursive build the graph
        if node is None: node = self.target_node
        operator = node.operator
        for name, dep_node in node.operands.items():
            dep_node.outdegree += 1
            self.build(dep_node)
        return self

    def _merge_elementwise(self, node):
        """element-wise ops (unary or binary) can be merged, thus reduce kernel calls. Consider the following computational graph.
        `a = b + c; d = a * e; ret = exp(d)`
        It has three element-wise kernel ops, i.e. add, mul and exp. We can merge that into a single kernel call, i.e. exp((b+c)*e).
        """
        operands = {}
        operator = node.operator
        for var_name, dep_node in node.operands.items():
            if dep_node.is_lazy:
                self._merge_elementwise(dep_node)
                if operator["type"] == "elementwise" and dep_node.operator["type"] == "elementwise" and dep_node.outdegree == 1:
                    operands.update(dep_node.operands)
                    var_expr = f"({dep_node.operator['code']})"
                    operator["code"] = operator["code"].replace(var_name, var_expr)
                else:
                    operands[var_name] = dep_node
            else:
                if operator["type"] == "elementwise":
                    new_var_name = varnamegetter.get()
                    operands[new_var_name] = dep_node
                    operator["code"] = operator["code"].replace(var_name, new_var_name)
                else:
                    operands[var_name] = dep_node
        node.operator, node.operands = operator, operands

    def _simplify_arithmetic(self):
        pass

    def _operation_fusion(self):
        pass

    def optimize(self):
        if OPT1: self._merge_elementwise(node=self.target_node)

    def debug(self, node=None, depth=0):
        pass

    def visualize(self):
        color_map = {"reduce": "#ecc30b", "elementwise": "#84bcda",
                     "matmul": "#f37748", "contiguous": "#d56062"}
        def build_nx_graph(node, G):
            if node is None: return G
            nid = id(node)
            operator = node.operator
            if nid in G.nodes: return G
            G.add_node(nid)
            G.nodes[nid]["label"] = f"{node.shape}\nin={node.indegree},out={node.outdegree}\n{nid}\nis_lazy={node.is_lazy}"
            lazy = True
            if node.is_lazy:
                G.nodes[nid]["fillcolor"] = color_map[operator["type"]]
            else:
                G.nodes[nid]["fillcolor"] = "#ffffff"; lazy = False
            G.nodes[nid]["style"] = "filled, dashed" if not lazy else "filled"
            for name, subnode in node.operands.items():
                G = build_nx_graph(subnode, G)
                edge = (id(subnode), nid)
                if edge not in G.edges:
                    opname = operator["type"][:3].upper()
                    if "code" in operator: opname += ("_" + operator["code"])
                    G.add_edge(*edge, cnt=1, label=opname)
            return G
        G = nx.DiGraph()
        G = build_nx_graph(self.target_node, G)
        mode = "array"
        nx.drawing.nx_pydot.write_dot(G, f"/tmp/{mode}.dot")
        os.system(f"dot -Tsvg /tmp/{mode}.dot -o /tmp/{mode}.svg")
        print(f"[GRAPH] save to /tmp/{mode}.svg")

