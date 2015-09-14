from __future__ import print_function, division

import psycopg2
from psycopg2.extras import Json

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
            cwd=cwd).strip()


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


def parse_filters(filter_expr):
    filters = []
    filter_kwargs = {}

    for f in filter_expr.split(":"):
        f = f.strip()
        equal_ind = f.find("~")
        if equal_ind < 0:
            raise ValueError("invalid filter: %s" % f)

        fname = f[:equal_ind]
        fval = f[equal_ind+1:]
        filters.append(
            "text(run_properties->'%s')" % fname
            + " ILIKE " +
            "%%(%s)s" % fname)

        filter_kwargs[fname] = "%" + fval + "%"

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


def enumerate_runs(args):
    db_conn = get_db_connection(create_schema=True)

    run_class = import_class(args.run_class)

    from socket import gethostname
    host = gethostname()

    import sys

    with db_conn:
        with db_conn.cursor() as cur:

            batch_size = 20
            count = 0
            for ibatch, batch in enumerate(batch_up(
                    batch_size, (
                        (host, args.run_class, Json(run_props))
                        for run_props in run_class.enumerate_runs({})))):
                cur.executemany("INSERT INTO run ("
                        "creation_machine_name,"
                        "run_class,"
                        "run_properties,"
                        "state) values (%s,%s,%s,'waiting');",
                        batch)

                count += len(batch)

                if ibatch % 15 == 0:
                    sys.stdout.write("(%d jobs created)" % count)
                else:
                    sys.stdout.write(".")
                sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.write("%d jobs created altogether\n" % count)

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

    run_class = import_class(args.run_class)

    from socket import gethostname
    host = gethostname()

    import sys

    filters = [
            ("run_class = %(run_class)s"),
            ("state = 'waiting'"),
            ]
    filter_kwargs = {"run_class": args.run_class}

    if args.filter:
        f, fk = parse_filters(args.filter)
        filters.extend(f)
        filter_kwargs.update(fk)

    where_clause = " AND ".join(filters)

    while True:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                        "SELECT id, run_properties FROM run "
                        "WHERE " + where_clause + " " +
                        "OFFSET floor(random()*("
                        "   SELECT COUNT(*) FROM run "
                        "   WHERE " + where_clause + " " +
                        ")) LIMIT 1",
                        filter_kwargs)
                rows = list(cur)

                if not rows:
                    break

                (id_, run_props), = rows

                if not args.dry_run:
                    cur.execute("UPDATE run SET state = 'running' WHERE id = %s;",
                            (id_,))

        if args.verbose:
            print(id_, run_props)

        env_properties = None

        try:
            env_properties = run_class.get_env_properties(run_props)

            result = run_class.run(run_props)
            state = "complete"
        except UnableToRun:
            state = "waiting"
            result = None
        except Exception as e:
            from traceback import format_exc
            tb = format_exc()

            state = "error"
            result = {
                    "error": type(e).__name__,
                    "error_value": str(e),
                    "traceback": tb,
                    }

        if args.verbose:
            print("->", state, result)
            print("  ", env_properties)

        if not args.dry_run:
            with db_conn.cursor() as cur:
                if state != "waiting" or (state == "error" and not args.stop):
                    cur.execute(
                            "UPDATE run "
                            "SET (state, env_properties, "
                            "   state_time, state_machine_name, results) "
                            "= (%(new_state)s, %(env_properties)s, "
                            "   current_timestamp, %(host)s, %(result)s) "
                            "WHERE id = %(id)s AND state = 'running';",
                            {"id": id_, "env_properties": Json(env_properties),
                                "host": host, "result": Json(result),
                                "new_state": state})
                else:
                    cur.execute(
                            "UPDATE run SET state = 'waiting' "
                            "WHERE id = %(id)s AND state = 'running;",
                            {"id": id_})

        if args.stop and state == "error":
            print(tb, file=sys.stderr)
            break

# }}}


# {{{ console

def table_from_cursor(cursor):
    from pytools import Table

    if cursor.description is None:
        return None

    tbl = Table()
    tbl.add_row([column[0] for column in cursor.description])
    for row in cursor:
        tbl.add_row(row)
    return tbl


def make_disttune_symbols(db_conn):
    return {
            "__name__": "__console__",
            "__doc__": None,
            "db_conn": db_conn,
            "table_from_cursor": table_from_cursor,
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
                    cur.execute(args)
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

# }}}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers()

    parser_enum = subp.add_parser("enum")
    parser_enum.add_argument("run_class")
    parser_enum.set_defaults(func=enumerate_runs)

    parser_reset_running = subp.add_parser("reset-running")
    parser_reset_running.add_argument("--filter", metavar="prop~val:prop~val")
    parser_reset_running.set_defaults(func=reset_running)

    parser_run = subp.add_parser("run")
    parser_run.add_argument("run_class")
    parser_run.add_argument("--stop",
            help="stop on exceptions", action="store_true")
    parser_run.add_argument("-n", "--dry-run",
            help="do not modify database", action="store_true")
    parser_run.add_argument("-v", "--verbose", action="store_true")
    parser_run.add_argument("--filter", metavar="prop~val:prop~val")
    parser_run.set_defaults(func=run)

    parser_console = subp.add_parser("console")
    parser_console.set_defaults(func=console)

    args = parser.parse_args()
    args.func(args)

# vim: foldmethod=marker
