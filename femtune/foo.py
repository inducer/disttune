#!/home/rkirby/Code/envs/lp/bin/python
import pyopencl as cl
import loopy as lp
import numpy as np
import os
from referencekernels import poissonrefknlfixed as pknl

os.environ["PYOPENCL_CTX"] = "0"
#os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
ctx = cl.create_some_context()

dtype = np.float64
#dtype = np.float32

nbf = 10
sdim = 3
nqp = 8

knl = pknl(np.float64, nbf, sdim, nqp)

#knl = lp.set_options(knl, edit_cl=True)

stuff = lp.auto_test_vs_ref(knl, ctx, parameters={'nels': 2*10**5},
                            quiet=True)

print stuff
