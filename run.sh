OPENCL_MEM_POOL=1
OPT_CONSTANT_FOLDING=1
OPT_ELEMWISE_FUSION=1
OPT_VIEWOP_PRUNING=0
DEBUG=0
GRAPH=0
LAZY=1
BACKEND=opencl

pytest -s
#pytest -s test/test_jit.py -k test_minimal
#python3 examples/mnist/run.py --batch_size 4096 --eval 1 --num_ep 3

