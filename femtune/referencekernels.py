import loopy as lp
import numpy as np


def poissonactionrefknl(dtype):
    loop_bounds = "{[c, j, ell]: \
                    0 <= c < nels and \
                    0 <= i < nbf and \
                    0 <= j < nbf and \
                    0 <= ell < sdim and \
                    0 <= ell2 < sdim and \
                    0 <= ell3 < sdim and \
                    0 <= k < nqp}"
    kernel = """
    refgradu(qp, dir) := sum(j, DPsi[j, qp, dir])
    gradu(qp, dir) := sum(ell, DFinv[c, ell, dir] * refgradu(qp, ell))
    gradtildeu(qp, dir) := sum(ell2, DFinv[c, dir, ell2] * gradu(qp, dir))
    integrand(c, bf, qp) := J[c] * sum(ell3, gradtildeu(qp, ell3) * DPsi[bf, qp, ell3]
    Aelu[c, i] = sum(k, integrand(i, k))
"""

    args = [lp.ValueArg("nels", np.int),
            lp.GlobalArg("Aelu", dtype, shape=lp.auto),
            lp.GlobalArg("DPsi", dtype, shape=lp.auto),
            lp.GlobalArg("DFinv", dtype, shape=lp.auto),
            lp.GlobalArg("J", dtype, shape=lp.auto),
            lp.GlobalArg("w", dtype, shape=lp.auto)]

    knl = lp.make_kernel(loop_bounds,
                         kernel,
                         args,
                         assumptions="nels>=1 and nbf >= 1 and nels mod 4 = 0")

    knl = lp.set_loop_priority(knl, ["c", "j", "i", "k"])
    knl = lp.fix_parameters(knl, nbf=nbf, sdim=sdim, nqp=nqp)

    

def poissonrefknlfixed(dtype, nbf, sdim, nqp):
    loop_bounds = "{ [c,i,j,k,ell,ell2]: \
                     0 <= c < nels and \
                     0 <= i < nbf and \
                     0 <= j < nbf and \
                     0 <= k < nqp and \
                     0 <= ell < sdim and \
                     0 <= ell2 < sdim }"

    kernel = \
        """
    dpsi(bf,k0,dir) := sum(ell2, DFinv[c,ell2,dir] * DPsi[bf,k0,ell2] )
    Ael[c,i,j] = Ael[c,i,j] + J[c] * w[k] * sum(ell, dpsi(i,k,ell) * dpsi(j,k,ell))"""
    args = [lp.ValueArg("nels", np.int),
            lp.GlobalArg("Ael", dtype, shape=lp.auto),
            lp.GlobalArg("DPsi", dtype, shape=lp.auto),
            lp.GlobalArg("DFinv", dtype, shape=lp.auto),
            lp.GlobalArg("J", dtype, shape=lp.auto),
            lp.GlobalArg("w", dtype, shape=lp.auto)]

    knl = lp.make_kernel(loop_bounds,
                         kernel,
                         args,
                         assumptions="nels>=1 and nbf >= 1 and nels mod 4 = 0")

    knl = lp.set_loop_priority(knl, ["c", "j", "i", "k"])
    knl = lp.fix_parameters(knl, nbf=nbf, sdim=sdim, nqp=nqp)

    return knl

def poissonrefknl(dtype):
    loop_bounds = "{ [c,i,j,k,ell,ell2]: \
                     0 <= c < nels and \
                     0 <= i < nbf and \
                     0 <= j < nbf and \
                     0 <= k < nqp and \
                     0 <= ell < sdim and \
                     0 <= ell2 < sdim }"

    kernel = \
        """
    dpsi(bf,k0,dir) := sum(ell2, DFinv[c,ell2,dir] * DPsi[bf,k0,ell2] )
    Ael[c,i,j] = Ael[c,i,j] + J[c] * w[k] * sum(ell, dpsi(i,k,ell) * dpsi(j,k,ell))"""
    args = [lp.ValueArg("nels", np.int),
            lp.GlobalArg("Ael", dtype, shape=lp.auto),
            lp.GlobalArg("DPsi", dtype, shape=lp.auto),
            lp.GlobalArg("DFinv", dtype, shape=lp.auto),
            lp.GlobalArg("J", dtype, shape=lp.auto),
            lp.GlobalArg("w", dtype, shape=lp.auto)]

    knl = lp.make_kernel(loop_bounds,
                         kernel,
                         args,
                         assumptions="nels>=1 and nbf >= 1 and nels mod 4 = 0")

    knl = lp.set_loop_priority(knl, ["c", "j", "i", "k"])
    return knl

	
def curlcurlrefknl(dtype):
    loop_bounds = "{ [c,i,j,k,ell,ell2]: \
                     0 <= c < nels and \
                     0 <= i < nbf and \
                     0 <= j < nbf and \
                     0 <= k < nqp and \
                     0 <= ell < sdim and \
                     0 <= ell2 < sdim }"

    kernel = \
        """
    dpsi(bf,k0,dir) := sum(ell2, DFinv[c,ell2,dir] * DPsi[bf,k0,ell2] )
    Ael[c,i,j] = Ael[c,i,j] + w[k] * sum(ell, dpsi(i,k,ell) * dpsi(j,k,ell)) / J[c]"""
    args = [lp.ValueArg("nels", np.int),
            lp.GlobalArg("Ael", dtype, shape=lp.auto),
            lp.GlobalArg("DPsi", dtype, shape=lp.auto),
            lp.GlobalArg("DFinv", dtype, shape=lp.auto),
            lp.GlobalArg("J", dtype, shape=lp.auto),
            lp.GlobalArg("w", dtype, shape=lp.auto)]

    knl = lp.make_kernel(loop_bounds,
                         kernel,
                         args,
                         assumptions="nels>=1 and nbf >= 1 and nels mod 4=0")

    knl = lp.set_loop_priority(knl, ["c", "j", "i", "k"])
    return knl


def curlcurlrefknlfixed(dtype, nbf, sdim, nqp):
    loop_bounds = "{ [c,i,j,k,ell,ell2]: \
                     0 <= c < nels and \
                     0 <= i < nbf and \
                     0 <= j < nbf and \
                     0 <= k < nqp and \
                     0 <= ell < sdim and \
                     0 <= ell2 < sdim }"

    kernel = \
        """
    dpsi(bf,k0,dir) := sum(ell2, DFinv[c,ell2,dir] * DPsi[bf,k0,ell2] )
    Ael[c,i,j] = Ael[c,i,j] + w[k] * sum(ell, dpsi(i,k,ell) * dpsi(j,k,ell)) / J[c]"""
    args = [lp.ValueArg("nels", np.int),
            lp.GlobalArg("Ael", dtype, shape=lp.auto),
            lp.GlobalArg("DPsi", dtype, shape=lp.auto),
            lp.GlobalArg("DFinv", dtype, shape=lp.auto),
            lp.GlobalArg("J", dtype, shape=lp.auto),
            lp.GlobalArg("w", dtype, shape=lp.auto)]

    knl = lp.make_kernel(loop_bounds,
                         kernel,
                         args,
                         assumptions="nels>=1 and nbf >= 1 and nels mod 4 = 0")

    knl = lp.set_loop_priority(knl, ["c", "j", "i", "k"])
    knl = lp.fix_parameters(knl, nbf=nbf, sdim=sdim, nqp=nqp)

    return knl
