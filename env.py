import os

DEBUG = int(os.getenv("DEBUG", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
LAZY = int(os.getenv("LAZY", "0"))
BACKEND = os.getenv("BACKEND", "opencl")
OPT_MERGE_ELEMWISE = int(os.getenv("OPT_MERGE_ELEMWISE", "0"))  # graph optimization: merge elementwise ops
INSTANT_VIEWOP2 = int(os.getenv("INSTANT_VIEWOP2", "0"))

assert BACKEND in ("numpy", "opencl", "cuda"), f"backend {BACKEND} not supported!"

if LAZY:
    assert BACKEND in ("opencl",), f"currently lazy mode only support opencl backend!"

