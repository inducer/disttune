from __future__ import print_function, division
import numpy as np

from disttune import RunBase


class Run(RunBase):
    @classmethod
    def enumerate_runs(cls, options):
        for it_type in ["list", "range"]:
            for split_factor in [10, 100, 1000, 10000]:
                yield {"it_type": it_type, "split_factor": split_factor}

    @classmethod
    def get_env_properties(cls, run_props):
        result = super(Run, cls).get_env_properties(run_props)

        result.update({
                "numpy_ver": np.version.version,
                })

        return result

    @classmethod
    def run(cls, run_props):
        it_type = run_props["it_type"]

        from time import time

        n = 10**6
        n_inner = run_props["split_factor"]
        n_outer = n // n_inner

        t_start = time()

        if it_type == "list":
            innerlist = list(range(n_inner))
            for i_outer in list(range(n_outer)):
                for i_inner in innerlist:
                    pass

        elif it_type == "range":

            for i_outer in range(n_outer):
                for i_inner in range(n_inner):
                    pass

        return {"t_elapsed": time() - t_start}
