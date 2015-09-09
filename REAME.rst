FEM autotuning
==============

Create jobs for an autotuning run::

    python -m disttune enum femtune.inner_outer_v1.Run

The dotted name ``femtune.inner_outer_v1.Run`` is actually the name of the
class that's driving the run.

Actually run jobs::

    python -m disttune run --stop femtune.inner_outer_v1.Run -v --filter "cl_platform~intel"

The filter expression can be used to limit what jobs are run.

Run interactive Python shell with database connection:

    python -m disttune console
    >>> .q select * from run;
