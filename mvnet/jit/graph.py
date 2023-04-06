import os
from collections import defaultdict

import networkx as nx

from mvnet.backend.base import ElemwiseOps, ProcessingOps, ReduceOps, ViewOps
from mvnet.utils.helper import varnamegetter


class GraphOptimizer:
  def __init__(self, root):
    assert root.is_lazy
    self.root = root
    varnamegetter.reset()

  def _rename_operands(self, root):
    def rename_operands(node):
      operands = {}
      for name, dep_node in node.op_info.operands.items():
        if not visited[id(dep_node)]:
          rename_operands(dep_node)
        new_name = name_dict[id(dep_node)]
        operands[new_name] = dep_node
        if type(node.op_info.operator) is ElemwiseOps:
          node.op_info.code = node.op_info.code.replace(name, new_name)
      node.op_info.operands = operands
      visited[id(node)] = True

    visited = defaultdict(bool)
    name_dict = defaultdict(varnamegetter.get)
    rename_operands(root)

  def _constant_folding(self, root):
    def constant_folding(node):
      if node.constant_value is not None:
        return True
      dep_is_const = []
      for name, dep_node in node.op_info.operands.items():
        if id(dep_node) not in cache:
          flag = constant_folding(dep_node)
          cache[id(dep_node)] = flag
        dep_is_const.append(cache[id(dep_node)])

      is_const = False
      if any(dep_is_const):
        if isinstance(node.op_info.operator, ViewOps):
          dep_node = next(iter(node.op_info.operands.values()))
          node.to_constant(dep_node.constant_value)
          is_const = True
        if isinstance(node.op_info.operator, ElemwiseOps):
          if all(dep_is_const) and len(dep_is_const) > 1:  # NOTE: skip unary ops
              expr = node.op_info.code
              for name, dep_node in node.op_info.operands.items():
                  expr = expr.replace(name, f"{dep_node.constant_value:.15f}f")
              node.to_constant(eval(expr.replace("f", "")))
              is_const = True
      return is_const

    cache = {}
    constant_folding(root)

  def _elemwise_fusion(self, root):
    def elemwise_fusion(node):
      for name in list(node.op_info.operands):
        dep_node = node.op_info.operands[name]
        if not dep_node.is_lazy:
          continue
        if not visited[id(dep_node)]:
          elemwise_fusion(dep_node)
        if type(node.op_info.operator) is ElemwiseOps and \
                type(dep_node.op_info.operator) is ElemwiseOps and \
                node.c_contiguous and dep_node.c_contiguous and \
                outdegree[id(dep_node)] == 1:
          node.op_info.operands.pop(name)
          node.op_info.operands.update(dep_node.op_info.operands)
          node.op_info.code = node.op_info.code.replace(name, f"({dep_node.op_info.code})")
      visited[id(node)] = True

    def update_outdegree(node):
      if visited[id(node)]: return
      for name, dep_node in node.op_info.operands.items():
        outdegree[id(dep_node)] += 1
        update_outdegree(dep_node)
      visited[id(node)] = True

    outdegree = defaultdict(int)
    visited = defaultdict(bool)
    update_outdegree(root)
    visited = defaultdict(bool)
    elemwise_fusion(root)

  def _viewop_pruning(self, root):
    def viewop_pruning(node):
      for name, dep_node in node.op_info.operands.items():
        if not visited[id(dep_node)]:
          viewop_pruning(dep_node)
        if type(node.op_info.operator) is ViewOps:
          node.op_info = dep_node.op_info
          node.constant_value = dep_node.constant_value
          if not dep_node.is_lazy and dep_node.constant_value is None:
              node.buffer = dep_node.buffer
      visited[id(node)] = True
    visited = defaultdict(bool)
    viewop_pruning(root)

  def _elemwise_processing_fusion(self, root):
    def elemwise_processing_fusion(node):
      dep_types = defaultdict(list)
      for name, dep_node in node.op_info.operands.items():
        if not visited[id(dep_node)]:
          elemwise_processing_fusion(dep_node)
        dep_types[type(dep_node.op_info.operator)].append((name, dep_node))
      if type(node.op_info.operator) is ElemwiseOps and \
              outdegree[id(dep_node)] == 1 and \
              len(dep_types[ProcessingOps]) == 1 and len(dep_types[ReduceOps]) == 0 and \
              all([not v.is_lazy for k, v in dep_types[ElemwiseOps]]):
        name, proc_dep = dep_types[ProcessingOps][0]
        extra_code = node.op_info.code.replace(name, "acc")
        extra_operands = {**dict(dep_types[ElemwiseOps]), **dict(dep_types[type(None)])}
        node.op_info = proc_dep.op_info
        node.op_info.args["extra"] = {"operands": extra_operands, "code": extra_code}
      visited[id(node)] = True

    def update_outdegree(node):
      if visited[id(node)]: return
      for name, dep_node in node.op_info.operands.items():
        outdegree[id(dep_node)] += 1
        update_outdegree(dep_node)
      visited[id(node)] = True

    outdegree = defaultdict(int)
    visited = defaultdict(bool)
    update_outdegree(root)
    visited = defaultdict(bool)
    elemwise_processing_fusion(root)

  def visualize(self, root, graph_name):
    colors = {ReduceOps: "#ecc30b", ElemwiseOps: "#84bcda", ProcessingOps: "#f37748", ViewOps: "#e5e5e5"}
    def build_graph(node, G):
      if node is None: return G
      if id(node) in G.nodes: return G
      G.add_node(id(node))
      label = (f"{node.shape}\n{node.strides}\n{id(node)}\nC:{int(node.c_contiguous)} F:{int(node.f_contiguous)}")
      if node.constant_value is not None:
        label += f"\nCONSTANT={node.constant_value}"
      if node.op_info.operator is not None:
        label += f"\n{node.op_info.operator.name}"
        if hasattr(node.op_info, "code"):
          label += f"\n{node.op_info.code}"
      G.nodes[id(node)]["label"] = label
      G.nodes[id(node)]["shape"] = "box" if node.constant_value is None else "ellipse"
      G.nodes[id(node)]["style"] = "filled, dashed" if node.is_lazy else "filled"
      G.nodes[id(node)]["fillcolor"] = colors.get(type(node.op_info.operator), "#ffffff")
      for name, subnode in node.op_info.operands.items():
        G = build_graph(subnode, G)
        edge = (id(subnode), id(node))
        if edge not in G.edges:
          G.add_edge(*edge, cnt=1, label=name)
      return G
    G = build_graph(root, G=nx.DiGraph())
    nx.drawing.nx_pydot.write_dot(G, f"/tmp/{graph_name}.dot")
    os.system(f"dot -Tsvg /tmp/{graph_name}.dot -o /tmp/{graph_name}.svg")
    print(f"[GRAPH] save to /tmp/{graph_name}.svg")

  def count(self, root):
    def count_node(node):
      cnt = 0
      for name, dep_node in node.op_info.operands.items():
        if not visited[id(dep_node)]:
          cnt += count_node(dep_node)
        cnt += 1
      visited[id(node)] = True
      return cnt
    visited = defaultdict(bool)
    return count_node(root)
