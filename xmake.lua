add_requires("doctest")
add_rules("mode.debug", "mode.release")

-- Custom rule to generate asm and populate template
rule("mont-gen-asm")
    set_extensions(".template")
    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        batchcmds:show_progress(opt.progress, '${color.build.object}templating from %s', sourcefile)
        batchcmds:execv("python3 mont/src/gen_asm.py", {sourcefile, target:targetfile()})
    end)
    on_link(function (target) end)

target("mont.cuh")
    add_files("mont/src/*.template")
    add_rules("mont-gen-asm")
    set_targetdir("mont/src")

target("test-mont")
    set_languages(("c++17"))
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_files("mont/tests/main.cu")
    add_packages("doctest")

target("bench-mont")
    set_languages(("c++17"))
    add_cugencodes("native")
    add_options("-lineinfo")
    add_options("--expt-relaxed-constexpr")
    add_files("mont/tests/bench.cu")

target("bench-mont0")
    add_deps("mont.cuh")
    add_options("-lineinfo")
    add_files("mont/tests/bench0.cu")

target("cuda_msm")
    set_kind("static")
    add_values("cuda.build.devlink", true)

    set_languages(("c++17"))
    
    add_files("wrapper/msm/c_api/msm_c_api.cu")
    add_headerfiles("wrapper/msm/c_api/*.h")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")

    set_targetdir("lib")

task("sync-epcc")
    on_run(function ()
        import ("core.base.option")

        os.runv("rsync -av . ".. option.get('target') ..":~/zksnark --exclude-from .gitignore", {
            stdout = 1
        ,   stderr = 2
        })
    end)
    set_menu {
        usage = "xmake sync-epcc"
    ,   description = "Synchronize the source code to EPCC node"
    ,   options =
        {
            {'t', 'target', 'kv', nil, 'Name of target node in SSH config'}
        }
    }

includes("runtime")
includes("ntt")
includes("wrapper")
-- includes("poly")
includes("msm")