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
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_deps("mont.cuh")
    add_files("mont/tests/main.cu")
    add_packages("doctest")

target("bench-mont")
    add_deps("mont.cuh")
    add_files("mont/tests/bench.cu")

target("test-bn254")
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_deps("mont.cuh")
    add_files("msm/tests/bn254.cu")
    add_packages("doctest")

target("test-msm")
    if is_mode("debug") then
        set_symbols("debug")
    end
    set_optimize("fastest")
    add_deps("mont.cuh")
    add_files("msm/tests/msm.cu")

target("test-ntt-int")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_deps("mont.cuh")
    add_headerfiles("NTT/src/*.cuh")
    add_files("NTT/tests/test-int.cu")
    add_cugencodes("native")

target("test-ntt-big")
    local project_root = os.projectdir()
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_deps("mont.cuh")
    add_files("NTT/tests/test-big.cu")
    add_headerfiles("NTT/src/*.cuh")
    add_cugencodes("native")
    add_packages("doctest")
    add_defines("PROJECT_ROOT=\"" .. project_root .. "\"")

target("cuda_ntt")
    set_kind("static")
    add_values("cuda.build.devlink", true)
    if is_mode("debug") then 
        set_symbols("debug")
    end

    set_languages(("c++20"))
    add_deps("mont.cuh")
    
    add_files("wrapper/NTT/c_api/ntt_c_api.cu")
    add_headerfiles("wrapper/NTT/c_api/*.h")
    add_headerfiles("NTT/src/*.cuh")
    add_cugencodes("native")

    set_targetdir("lib")

task("sync-epcc")
    on_run(function ()
        os.runv("rsync -av . epcc4090:~/zksnark --exclude-from .gitignore", {
            stdout = 1
        ,   stderr = 2
        })
    end)
    set_menu {
        usage = "xmake sync-epcc"
    ,   description = "Synchronize the source code to EPCC node"
    }
