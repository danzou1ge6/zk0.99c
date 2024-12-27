target("test-bn254")
    if is_mode("debug") then
        set_symbols("debug")
    end
    set_languages(("c++20"))
    add_files("bn254.cu")
    add_cugencodes("native")
    add_packages("doctest")

target("test-msm")
    set_languages(("c++20"))
    add_files("msm.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")

target("test-msm0")
    set_languages(("c++20"))
    add_files("msm.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")
    add_defines("NO_LAZY_MODULO")