Distributed Autotuning
======================

For all of this, you'll need authentication details for a PostgreSQL server,
in a bunch of environment variables. Perhaps you might set those like this::

    export TUNE_DBHOST=tuning-database.example.com
    export TUNE_DBNAME=disttune
    export TUNE_DBUSER=sampleuse
    export TUNE_DBPASSWORD=Yeizah2u

Create jobs for an autotuning run::

    python -m disttune enum exampletune.Run

The dotted name ``exampletune.Run`` is actually the name of the
class that's driving the run.

Actually run jobs::

    python -m disttune run --stop -v --filter run_class=exampletune.Run "it_type~list"

The filter expression can be used to limit what jobs are run.

Run interactive Python shell with database connection::

    python -m disttune console
    >>> .q select * from run;
