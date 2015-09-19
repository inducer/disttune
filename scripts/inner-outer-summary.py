p("select state_machine_name as machine, rp.cl_platform as cl_plat, rp.sd as sd, rp.deg as deg, rp.prec as prec, "
        "min(res.elapsed_wall::float) as min, "
        "max(res.elapsed_wall::float) as max, "
        "max(res.elapsed_wall::float)/min(res.elapsed_wall::float) as factor, "
        "count(*) from run "
        "where '[\"andreas\"]'::jsonb <@ rp..tags and state='complete' "
        "group by state_machine_name, rp.cl_platform, rp.sd, rp.deg, rp.prec "
        "order by rp.sd, rp.deg, rp.prec, state_machine_name, rp.cl_platform;")
