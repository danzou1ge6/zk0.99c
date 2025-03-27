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
    add_files("../src/fast_compile/*.cu")
    add_cugencodes("native")

target("msm-ppzsnark-dumps")
    set_languages(("c++20"))
    add_files("msm_ppzsnark_dumps.cu")
    add_files("../src/fast_compile/*.cu")
    add_cugencodes("native")