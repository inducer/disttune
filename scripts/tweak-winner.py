from __future__ import print_function
from psycopg2.extras import Json
import pyopencl as cl
import loopy as lp
import numpy as np


sd = 3
deg = 2
prec = "float64"
print("sd", sd, "deg", deg)

rows = q(
    "select id, run_class, env_properties, run_properties, results from run "
    "where "
    "%(tags)s <@ rp..tags "
    "and state = 'complete' "
    "and run_class = %(run_class)s "
    "and cast(rp.sd as int) = %(sd)s "
    "and cast(rp.deg as int) = %(deg)s "
    "and rp.prec = %(prec)s "
    "and rp.cl_platform ILIKE %(cl_plat)s "
    "order by cast(res.elapsed_wall as float) "
    "limit 1",
    sd=sd, deg=deg, prec=prec, run_class="femtune.inner_outer_v1.Run",
    cl_plat="%nvi%", tags=Json(["andreas"])
    )

for rid, run_class_name, env, run_props, res in rows:
    print(rid, res["elapsed_wall"], run_props)

from disttune import import_class, get_cl_device
run_class = import_class(run_class_name)

dev = get_cl_device(run_props)
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # noqa

refknl, knl = run_class.get_loopy_kernel(run_props, dev)

if 0:
    print(knl)

time = res["elapsed_wall"]

parameter_dict = {"nels": run_props["nels"]}

op_poly = lp.get_op_poly(knl)
op_poly_ev = {
    k: 1e-9*v.eval_with_dict(parameter_dict)/time
    for k, v in op_poly.items()}
print("GFLOPS/S", op_poly_ev[np.dtype(prec)])

gmem_poly = lp.sum_mem_access_to_bytes(lp.get_gmem_access_poly(knl))
gmem_poly_ev = {
    k: 1e-9*v.eval_with_dict(parameter_dict)/time
    for k, v in gmem_poly.items()}
print("GMEM [GB/s]")
print(lp.stringify_stats_mapping(gmem_poly_ev))

if 1:
    print(knl)
    #knl = lp.add_prefetch(knl, "DFinv[c_inner+c_outer*8,:,:]")
    knl = lp.add_prefetch(knl, "DPsi[:,:,:]", default_tag=None)
    print(knl)

    res2 = lp.auto_test_vs_ref(
            knl, ctx, parameters={'nels': run_props["nels"]},
            quiet=True)

    print(res2)

if 1:
    knl = lp.preprocess_kernel(knl)
    code, _ = lp.generate_code(knl)
    print(code)

    if 0:  # print ptx
        prg = cl.Program(ctx, code).build()
        print(prg.binaries[0])

