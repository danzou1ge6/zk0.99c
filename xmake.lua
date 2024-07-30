add_requires("doctest")

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
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_deps("mont.cuh")
    add_files("mont/tests/*.cu")
    add_packages("doctest")

target("test-bn254")
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_deps("mont.cuh")
    add_files("msm/tests/bn254.cu")
    add_packages("doctest")

task("sync-epcc")
    on_run(function ()
        os.runv("rsync -av --delete . epcc4090:~/zksnark --exclude-from .gitignore", {
            stdout = 1
        ,   stderr = 2
        })
    end)
    set_menu {
        usage = "xmake sync-epcc"
    ,   description = "Synchronize the source code to EPCC node"
    }
