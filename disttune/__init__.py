from __future__ import print_function, division

import psycopg2
from psycopg2.extras import Json
from psycopg2.extensions import TransactionRollbackError

import code


# {{{ database setup

SCHEMA = ["""
CREATE TYPE job_state AS ENUM (
    'waiting',
    'running',
    'error',
    'complete');
""",
"""
CREATE TABLE run (
    id                     SERIAL PRIMARY KEY,

    -- filled on generation
    creation_time          TIMESTAMP DEFAULT current_timestamp,
    creation_machine_name  VARCHAR(255),
    run_class              VARCHAR(255),
    run_properties         JSONB,

    -- updated throughout
    state                  job_state,

    -- filled on run
    env_properties         JSONB,
    state_time             TIMESTAMP,
    state_machine_name     VARCHAR(255),
    results                JSONB
    );
"""]


def try_create_schema(db_conn):
    with db_conn:
        with db_conn.cursor() as cur:
            try:
                cur.execute(SCHEMA[0])
            except psycopg2.ProgrammingError as e:
                if "already exists" in str(e):
                    return
                else:
                    raise

            for s in SCHEMA[1:]:
                cur.execute(s)


def get_db_connection(create_schema):
    import os
    db_host = os.environ.get("TUNE_DBHOST", None)
    db_name = os.environ.get("TUNE_DBNAME", "tune_db")

    import getpass
    db_user = os.environ.get("TUNE_DBUSER", getpass.getuser())
    db_password = os.environ.get("TUNE_DBPASSWORD")

    db_conn = psycopg2.connect(
            host=db_host, dbname=db_name, user=db_user,
            password=db_password, sslmode="require")

    if create_schema:
        try_create_schema(db_conn)

    db_conn.set_session(isolation_level="serializable")

    return db_conn

# }}}


def get_git_rev(module=None):
    if module:
        from importlib import import_module
        mod = import_module(module)
        from os.path import dirname
        cwd = dirname(mod.__file__)
    else:
        from os import getcwd
        cwd = getcwd()

    from subprocess import check_output
    return check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd).strip().decode()


class UnableToRun(Exception):
    """A temporary run failure. Will not mark the job as permanently failed."""


class RunBase(object):
    @classmethod
    def enumerate_runs(cls, options):
        raise NotImplementedError()

    @classmethod
    def get_env_properties(cls, run_props):
        import sys
        return {
                "python_version": sys.version
                }

    @classmethod
    def run(cls, run_props):
        pass


class ClassNotFoundError(RuntimeError):
    pass


def import_class(name):
    components = name.split('.')

    if len(components) < 2:
        # need at least one module plus class name
        raise ClassNotFoundError(name)

    module_name = ".".join(components[:-1])
    try:
        mod = __import__(module_name)
    except ImportError:
        raise ClassNotFoundError(name)

    for comp in components[1:]:
        try:
            mod = getattr(mod, comp)
        except AttributeError:
            raise ClassNotFoundError(name)

    return mod


# {{{ cl utilities

def enumerate_distinct_cl_devices(exclude_platforms=[]):
    import pyopencl as cl

    seen = set()
    for plat in cl.get_platforms():
        if any(
                excl.lower() in plat.name.lower()
                or
                excl.lower() in plat.vendor.lower()
                for excl in exclude_platforms):
            continue

        for dev in plat.get_devices():
            pd = (plat.name, dev.name)
            if pd in seen:
                continue

            seen.add(pd)
            yield {
                    "cl_platform": pd[0],
                    "cl_device": pd[1],
                    }


class CLDeviceNotFound(UnableToRun):
    pass


def get_cl_device(run_props):
    import pyopencl as cl

    for plat in cl.get_platforms():
        if plat.name == run_props["cl_platform"]:
            for dev in plat.get_devices():
                if dev.name == run_props["cl_device"]:
                    return dev

    raise CLDeviceNotFound(str(run_props["cl_platform"]) + ", "
                           + str(run_props["cl_device"]))


def get_cl_properties(dev):
    plat = dev.platform
    return {
        "pyopencl_git_rev": get_git_rev("pyopencl"),

        "cl_platform_vendor": plat.vendor,
        "cl_platform_version": plat.version,
        "cl_device_vendor": dev.vendor,
        "cl_device_version": dev.version,
        "cl_device_extensions": dev.extensions,
        "cl_device_address_bits": dev.address_bits,
        }

# }}}


def parse_filters(filter_args):
    filters = []
    filter_kwargs = {}

    for f in filter_args:
        f = f.strip()
        op_ind = max(f.find("~"), f.find("="))
        op = f[op_ind]

        if op_ind < 0:
            raise ValueError("invalid filter: %s" % f)

        fname = f[:op_ind]
        fval = f[op_ind+1:]

        if fname in [
                "id",
                "run_class",
                "creation_machine_name",
                ]:
            lhs = "text(%s)" % fname

        else:
            lhs = "run_properties->>'%s'" % fname

        from random import choice
        f_kwarg_name = fname
        while f_kwarg_name in filter_kwargs:
            f_kwarg_name += choice("0123456789")

        if op == "~":
            filters.append(
                # case insensitive regex
                lhs + " ~* " +
                "%%(%s)s" % f_kwarg_name)
            filter_kwargs[f_kwarg_name] = ".*" + fval + ".*"
        elif op == "=":
            filters.append(lhs + "=" + "%%(%s)s" % f_kwarg_name)
            filter_kwargs[f_kwarg_name] = fval
        else:
            raise ValueError("invalid operand")

    return filters, filter_kwargs


# {{{ enumerate

def batch_up(n, iterator):
    batch = []
    for i in iterator:
        batch.append(i)
        if len(batch) >= n:
            yield batch
            del batch[:]

    if batch:
        yield batch


def limit_iterator(max_count, it):
    count = 0
    for item in it:
        yield item

        count += 1
        if max_count is not None and count >= max_count:
            return


def enumerate_runs(args):
    db_conn = get_db_connection(create_schema=True)

    run_class = import_class(args.run_class)

    from socket import gethostname
    host = gethostname()

    enum_options = {}
    if args.options:
        for s in args.options:
            s = s.strip()
            equal_ind = s.find("=")
            if equal_ind < 0:
                raise ValueError("invalid enum argument: %s" % s)

            aname = s[:equal_ind]
            aval = s[equal_ind+1:]
            enum_options[aname] = aval

    def get_enum_iterator():
        it = run_class.enumerate_runs(enum_options)
        it = limit_iterator(args.limit, it)
        return it

    total_count = 0

    print("counting...")

    for ijob, run_props in enumerate(get_enum_iterator()):
        if ijob % 10000 == 0 and ijob:
            print("%d jobs, still counting..." % ijob)
        total_count += 1

    print("creating %d jobs..." % total_count)

    def add_args(run_props):
        run_props = run_props.copy()
        if args.tags:
            run_props["tags"] = args.tags
        if enum_options:
            run_props["enum_options"] = enum_options

        return run_props

    with db_conn:
        with db_conn.cursor() as cur:

            from pytools import ProgressBar
            pb = ProgressBar("enumerate jobs", total_count)

            batch_size = 20
            count = 0
            for ibatch, batch in enumerate(batch_up(
                    batch_size, (
                        (host, args.run_class, Json(add_args(run_props)))
                        for run_props in get_enum_iterator()))):
                cur.executemany("INSERT INTO run ("
                        "creation_machine_name,"
                        "run_class,"
                        "run_properties,"
                        "state) values (%s,%s,%s,'waiting');",
                        batch)

                pb.progress(len(batch))
                count += len(batch)

            pb.finished()

            print("%d jobs created" % count)

# }}}


def reset_running(args):
    db_conn = get_db_connection(create_schema=True)

    filters = [
            ("state = 'running'"),
            ]
    filter_kwargs = {}

    if args.filter:
        f, fk = parse_filters(args.filter)
        filters.extend(f)
        filter_kwargs.update(fk)

    where_clause = " AND ".join(filters)

    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                    "UPDATE run SET state = 'waiting' "
                    "WHERE " + where_clause + ";",
                    filter_kwargs)


# {{{ run

def run(args):
    db_conn = get_db_connection(create_schema=True)

    from socket import gethostname
    host = gethostname()

    import sys

    filters = [("state = 'waiting'")]
    filter_kwargs = {}

    if args.filter:
        f, fk = parse_filters(args.filter)
        filters.extend(f)
        filter_kwargs.update(fk)

    where_clause = " AND ".join(filters)

    while True:
        try:
            # Start transaction for atomic state update.
            with db_conn:

                with db_conn.cursor() as cur:
                    cur.execute(
                            "SELECT id, run_class, run_properties FROM run "
                            "WHERE " + where_clause + " " +
                            "OFFSET floor(random()*("
                            "   SELECT COUNT(*) FROM run "
                            "   WHERE " + where_clause + " " +
                            ")) LIMIT 1",
                            filter_kwargs)
                    rows = list(cur)

                    if not rows:
                        break

                    (id_, run_class, run_props), = rows

                    if not args.dry_run:
                        cur.execute(
                                "UPDATE run SET state = 'running' WHERE id = %s;",
                                (id_,))

        except TransactionRollbackError:
            if args.verbose:
                print("Retrying job retrieval...")
            continue

        if args.verbose:
            print(75*"=")
            print(id_, run_class, run_props)
            print(75*"-")

        env_properties = None

        run_class = import_class(run_class)
        try:
            env_properties = run_class.get_env_properties(run_props)

            result = run_class.run(run_props)
            state = "complete"
        except UnableToRun:
            state = "waiting"
            result = None

            if args.verbose:
                print(75*"-")
                print("-> unable to run")
                from traceback import print_exc
                print_exc()

        except Exception as e:
            from traceback import format_exc
            tb = format_exc()

            if args.retry_on_error:
                state = "waiting"
                result = None

                disposition_msg = "error (will be retried)"

            else:
                state = "error"
                result = {
                        "error": type(e).__name__,
                        "error_value": str(e),
                        "traceback": tb,
                        }

                disposition_msg = "error (permanent)"

            if args.verbose:
                print(75*"-")
                print("->", disposition_msg)
                from traceback import print_exc
                print_exc()

        else:
            if args.verbose:
                print("->", state)
                print("  ", result)
                print("  ", env_properties)

        if not args.dry_run:
            while True:
                try:
                    # Start transaction. Otherwise we'll implicitly start a
                    # transaction that contains the rest of our run.
                    with db_conn:

                        with db_conn.cursor() as cur:
                            if state != "waiting" or (
                                    state == "error" and not args.stop_on_error):
                                cur.execute(
                                    "UPDATE run "
                                    "SET (state, env_properties, "
                                    "   state_time, state_machine_name, results) "
                                    "= (%(new_state)s, %(env_properties)s, "
                                    "   current_timestamp, %(host)s, %(result)s) "
                                    "WHERE id = %(id)s AND state = 'running';",
                                    {"id": id_,
                                        "env_properties": Json(env_properties),
                                        "host": host,
                                        "result": Json(result),
                                        "new_state": state})
                            else:
                                cur.execute(
                                        "UPDATE run SET state = 'waiting' "
                                        "WHERE id = %(id)s AND state = 'running';",
                                        {"id": id_})

                except TransactionRollbackError:
                    if args.verbose:
                        print("Retrying job result submission...")

                else:
                    break

        if args.stop_on_error and state == "error":
            print(tb, file=sys.stderr)
            break

# }}}


# {{{ shell

def table_from_cursor(cursor):
    from pytools import Table

    if cursor.description is None:
        return None

    tbl = Table()
    tbl.add_row([column[0] for column in cursor.description])
    for row in cursor:
        tbl.add_row(row)
    return tbl


def mangle_query(qry):
    import re
    qry, _ = re.subn(r"rp\.([a-zA-Z_0-9]+)", r"(run_properties->>'\1')", qry)
    qry, _ = re.subn(r"ep\.([a-zA-Z_0-9]+)", r"(env_properties->>'\1')", qry)
    qry, _ = re.subn(r"res\.([a-zA-Z_0-9]+)", r"(results->>'\1')", qry)
    qry, _ = re.subn(r"rp\.\.([a-zA-Z_0-9]+)", r"(run_properties->'\1')", qry)
    qry, _ = re.subn(r"ep\.\.([a-zA-Z_0-9]+)", r"(env_properties->'\1')", qry)
    qry, _ = re.subn(r"res\.\.([a-zA-Z_0-9]+)", r"(results->'\1')", qry)
    return qry


def make_disttune_symbols(db_conn):
    def q(qry, *arg_dicts, **extra_kwargs):
        args = {}
        args.update(extra_kwargs)
        for d in arg_dicts:
            args.update(d)

        cur = db_conn.cursor()
        cur.execute(mangle_query(qry), args)
        return cur

    return {
            "__name__": "__console__",
            "__doc__": None,
            "db_conn": db_conn,
            "q": q,
            "p": lambda qry: print(table_from_cursor(q(qry))),
            "table_from_cursor": table_from_cursor,
            "mangle_query": mangle_query,
            }


class DisttuneConsole(code.InteractiveConsole):
    def __init__(self, db_conn):
        self.db_conn = db_conn
        code.InteractiveConsole.__init__(self,
                make_disttune_symbols(db_conn))

        try:
            import numpy  # noqa
            self.runsource("import numpy as np")
        except ImportError:
            pass

        try:
            import matplotlib.pyplot  # noqa
            self.runsource("import matplotlib.pyplot as pt")
        except ImportError:
            pass
        except RuntimeError:
            pass

        try:
            import readline
            import rlcompleter  # noqa
            have_readline = True
        except ImportError:
            have_readline = False

        if have_readline:
            import os
            import atexit

            readline.set_history_length(-1)
            histfile = os.path.join(os.environ["HOME"], ".disttunehist")
            if os.access(histfile, os.R_OK):
                readline.read_history_file(histfile)
            atexit.register(readline.write_history_file, histfile)
            readline.parse_and_bind("tab: complete")

        self.last_push_result = False

    def push(self, cmdline):
        if cmdline.startswith("."):
            try:
                self.execute_magic(cmdline)
            except:
                import traceback
                traceback.print_exc()
        else:
            self.last_push_result = code.InteractiveConsole.push(self, cmdline)

        return self.last_push_result

    def execute_magic(self, cmdline):
        cmd_end = cmdline.find(" ")
        if cmd_end == -1:
            cmd = cmdline[1:]
            args = ""
        else:
            cmd = cmdline[1:cmd_end]
            args = cmdline[cmd_end+1:]

        if cmd == "help":
            print("""
Commands:
 .help        show this help message
 .q SQL       execute a (potentially mangled) query

Available Python symbols:
    db_conn: the database
    q(query_str): get database cursor for query_str
    dbprint(cursor): print result of cursor
    table_from_cursor(cursor)
""")
        elif cmd == "q":
            with self.db_conn:
                with self.db_conn.cursor() as cur:
                    cur.execute(mangle_query(args))
                    tbl = table_from_cursor(cur)
                    if tbl is not None:
                        print(tbl)

        else:
            print("invalid magic command")


def console(args):
    db_conn = get_db_connection(create_schema=False)

    import sys
    cons = DisttuneConsole(db_conn)
    cons.interact("Disttune running on Python %s\n"
            "Copyright (c) Andreas Kloeckner 2015\n"
            "Run .help to see help" % sys.version)


def script(args):
    db_conn = get_db_connection(create_schema=False)

    from os.path import abspath, dirname
    scriptdir = dirname(abspath(args.script))

    import sys
    sys.path.append(scriptdir)

    namespace = make_disttune_symbols(db_conn)

    with open(args.script, "rt") as s:
        script_contents = s.read()

    exec(compile(script_contents, args.script, 'exec'), namespace)

# }}}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers()

    parser_enum = subp.add_parser("enum")
    parser_enum.add_argument("run_class")
    parser_enum.add_argument("--options", metavar="KEY=VAL", nargs="*",
            help="specify options to be passed to enumerate_runs()")
    parser_enum.add_argument("--tags", metavar="TAG", nargs="*")
    parser_enum.add_argument("--limit", metavar="COUNT", type=int,
            help="create at most COUNT jobs")
    parser_enum.set_defaults(func=enumerate_runs)

    parser_reset_running = subp.add_parser("reset-running")
    parser_reset_running.add_argument(
            "--filter", metavar="prop=val or prop~val", nargs="*")
    parser_reset_running.set_defaults(func=reset_running)

    parser_run = subp.add_parser("run")
    parser_run.add_argument("--stop-on-error",
            help="stop execution on exceptions", action="store_true")
    parser_run.add_argument("--retry-on-error",
            help="if execution fails with an error, return run to 'waiting' status",
            action="store_true")
    parser_run.add_argument("-n", "--dry-run",
            help="do not modify database", action="store_true")
    parser_run.add_argument("-v", "--verbose", action="store_true")
    parser_run.add_argument(
            "--filter", metavar="prop=val or prop~val", nargs="*")
    parser_run.set_defaults(func=run)

    parser_console = subp.add_parser("console")
    parser_console.set_defaults(func=console)

    parser_script = subp.add_parser("script")
    parser_script.add_argument("script", metavar="SCRIPT.PY")
    parser_script.add_argument("script_args", metavar="ARG",
        nargs="*")
    parser_script.set_defaults(func=script)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_usage()
        import sys
        sys.exit(1)

    args.func(args)

# vim: foldmethod=marker
