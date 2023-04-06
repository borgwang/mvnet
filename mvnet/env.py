import os

DEBUG = int(os.getenv("DEBUG", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
LAZY = int(os.getenv("LAZY", "0"))
BACKEND = os.getenv("BACKEND", "opencl")

OPT_CONSTANT_FOLDING = int(os.getenv("OPT_CONSTANT_FOLDING", "0"))
OPT_ELEMWISE_FUSION = int(os.getenv("OPT_ELEMWISE_FUSION", "0"))
OPT_VIEWOP_PRUNING = int(os.getenv("OPT_VIEWOP_PRUNING", "0"))
OPT_ELEMWISE_PROCESSING_FUSION = int(os.getenv("OPT_ELEMWISE_PROCESSING_FUSION", "0"))

assert BACKEND in ("numpy", "opencl", "cuda"), f"backend {BACKEND} not supported!"

if LAZY:
  assert BACKEND in ("opencl",), "currently lazy mode only support opencl backend!"
