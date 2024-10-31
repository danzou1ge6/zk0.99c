target("test-ntt-int")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../../mont/src/*.cuh")
    add_headerfiles("../../ntt/src/*.cuh")
    add_files("../../ntt/tests/test-int.cu")
    add_cugencodes("native")

target("test-ntt-recompute")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../../mont/src/*.cuh")
    add_headerfiles("../../ntt/src/*.cuh")
    add_files("../../ntt/tests/test-recompute.cu")
    add_cugencodes("native")

target("test-ntt-big")
    local project_root = os.projectdir()
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../../mont/src/*.cuh")
    add_files("../../ntt/tests/test-big.cu")
    add_headerfiles("../../ntt/src/*.cuh")
    add_cugencodes("native")
    add_packages("doctest")
    add_defines("PROJECT_ROOT=\"" .. project_root .. "\"")

target("test-ntt-parallel")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../../mont/src/*.cuh")
    add_headerfiles("../../ntt/src/*.cuh")
    add_files("../../ntt/tests/test-parallel.cu")
    add_cugencodes("native")

target("test-ntt-transpose")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../src/inplace_transpose/cuda/*.cuh")
    add_headerfiles("../src/inplace_transpose/common/*.h")
    add_headerfiles("../src/inplace_transpose/common/*.cuh")
    add_files("../src/inplace_transpose/cuda/*.cu")
    add_files("../src/inplace_transpose/common/*.cpp")
    add_files("../src/inplace_transpose/common/*.cu")
    add_files("test-transpose.cu")
    add_cugencodes("native")

target("test-ntt-4step")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../../mont/src/*.cuh")
    add_headerfiles("../../ntt/src/*.cuh")
    add_files("../../ntt/tests/test-4step.cu")

    add_headerfiles("../src/inplace_transpose/cuda/*.cuh")
    add_headerfiles("../src/inplace_transpose/common/*.h")
    add_headerfiles("../src/inplace_transpose/common/*.cuh")
    add_files("../src/inplace_transpose/cuda/*.cu")
    add_files("../src/inplace_transpose/common/*.cpp")
    add_files("../src/inplace_transpose/common/*.cu")
    add_cugencodes("native")