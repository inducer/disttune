from __future__ import print_function, division

from disttune import RunBase
import numpy as np
import loopy as lp
import pyopencl as cl


def pdim_lagrange(sd, deg):
    if sd == 1:
        return deg+1
    elif sd == 2:
        return (deg+1)*(deg+2)//2
    elif sd == 3:
        return (deg+1)*(deg+2)*(deg+3)//6
    else:
        raise ValueError("Illegal spatial dimension")


def apply_options(knl, i_split, j_split, nbf):
    # j should be lowest axis
    num_parallel = 0
    num_work_items = 1
    if j_split != "serial":
        if j_split == "simple":
            if knl: knl = lp.tag_inames(knl, {"j": "l.0"})
            num_work_items *= nbf
        else:
            (inout, size) = j_split
            if inout == "inner":
                if knl: knl = lp.split_iname(knl, "j", size, inner_tag="l.0")
                num_work_items *= size
            else:
                if knl: knl = lp.split_iname(knl, "j", size, outer_tag="l.0")
                num_work_items *= (nbf / size)
        num_parallel += 1
    if i_split != "serial":
        ltag = "l.%d" % (num_parallel,)
        if i_split == "simple":
            if knl: knl = lp.tag_inames(knl, {"i": ltag})
            num_work_items *= nbf
        else:
            (inout, size) = i_split
            if inout == "inner":
                if knl: knl = lp.split_iname(knl, "i", size, inner_tag=ltag)
                num_work_items *= size
            else:
                if knl: knl = lp.split_iname(knl, "i", size, outer_tag=ltag)
                num_work_items *= (nbf / size)
        num_parallel += 1

    return (knl, num_parallel, num_work_items)


def apply_num_cells(knl, num_cells, num_parallel, is_cpu):
    if num_cells == 1:
        knl = lp.tag_inames(knl, {"c": "g.0"})
        if is_cpu:
            knl = lp.tag_inames(knl, {"ell": "unr", "ell2": "unr"})

    else:
        knl = lp.split_iname(
            knl, "c", num_cells,
            outer_tag="g.0",
            inner_tag="l.%d" % (num_parallel,))
        if is_cpu:
            knl = lp.tag_inames(
                knl, {"ell": "unr", "ell2": "unr"})

    return knl


def factors(n):
    result = [1]
    i = 2
    while i <= np.sqrt(n):
        if not n % i:
            result.append(i)
        i += 1
    result.append(n)
    return result


def get_nels(sd, dev):
    if dev.type & cl.device_type.CPU:
        nperdim = {2: 512, 3: 32}
    else:
        nperdim = {2: 512, 3: 32}

    if sd == 2:
        return 2*nperdim[2]**2
    elif sd == 3:
        return 6*nperdim[3]**3


class Run(RunBase):
    @classmethod
    def enumerate_runs(cls, options):
        from disttune import enumerate_distinct_cl_devices, get_cl_device

        for cl_dev_data in enumerate_distinct_cl_devices():
            dev = get_cl_device(cl_dev_data)

            if "cl_platform" in options:
                if options["cl_platform"].lower() not in dev.platform.name.lower():
                    continue

            max_wg = min(dev.max_work_group_size, 1024)

            for sd in [2, 3]:
                for deg in [1, 2, 3]:
                    nbf = pdim_lagrange(sd, deg)

                    bf_factors = factors(nbf)

                    split_options = ["serial", "simple"]
                    split_options.extend(
                        [("inner", f) for f in bf_factors[1:-1]])
                    split_options.extend(
                        [("outer", f) for f in bf_factors[1:-1]])

                    for i_split in split_options:
                        for j_split in split_options:
                            (_, num_parallel, num_work_items) \
                                    = apply_options(None, i_split, j_split, nbf)

                            nc_values = [1]

                            num_cells = 2

                            # now try multiple cells
                            while (num_cells * num_work_items < max_wg
                                    and num_cells < 15):
                                nc_values.append(num_cells)
                                num_cells += 1

                            for num_cells in nc_values:

                                from pyopencl.characterize import has_double_support
                                if has_double_support(dev):
                                    precs = ["float32", "float64"]
                                else:
                                    precs = ["float32"]

                                for prec in precs:
                                        run_props = {
                                                "prec": prec,
                                                "sd": sd,
                                                "deg": deg,
                                                "i_split": i_split,
                                                "j_split": j_split,
                                                "num_cells": num_cells,
                                                "nels": get_nels(sd, dev),
                                                }
                                        run_props.update(cl_dev_data)

                                        yield run_props

    @classmethod
    def get_env_properties(cls, run_props):
        result = super(Run, cls).get_env_properties(run_props)

        from disttune import get_git_rev, get_cl_device, get_cl_properties

        result.update({
                "numpy_ver": np.version.version,
                "loopy_git_rev": get_git_rev("loopy"),
                "femtune_git_rev": get_git_rev("femtune"),
                })

        dev = get_cl_device(run_props)
        result.update(get_cl_properties(dev))

        return result

    @classmethod
    def get_loopy_kernel(cls, run_props, dev):
        from femtune.referencekernels import poissonrefknl

        dtype = np.dtype(run_props["prec"])
        knl = poissonrefknl(dtype)

        sd = run_props["sd"]
        deg = run_props["deg"]
        i_split = run_props["i_split"]
        j_split = run_props["j_split"]
        num_cells = run_props["num_cells"]

        nbf = pdim_lagrange(sd, deg)
        nqp = deg**sd

        knl = lp.fix_parameters(knl, nbf=nbf, sdim=sd, nqp=nqp)
        refknl = knl

        nbf = pdim_lagrange(sd, deg)

        (knl, num_parallel, num_work_items) \
                = apply_options(knl, i_split, j_split, nbf)

        if num_cells is not None:
            knl = apply_num_cells(
                    knl, num_cells=num_cells,
                    num_parallel=num_parallel,
                    is_cpu=bool(dev.type & cl.device_type.CPU))

        return refknl, knl

    @classmethod
    def run(cls, run_props):
        from disttune import get_cl_device
        dev = get_cl_device(run_props)

        ctx = cl.Context([dev])
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # noqa

        refknl, knl = cls.get_loopy_kernel(run_props, dev)

        return lp.auto_test_vs_ref(
                knl, ctx, parameters={'nels': run_props["nels"]},
                quiet=True)
