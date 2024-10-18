target("cuda_ntt")
    set_kind("static")
    add_values("cuda.build.devlink", true)
    if is_mode("debug") then 
        set_symbols("debug")
    end

    set_languages(("c++20"))
    add_headerfiles("../../mont/src/*.cuh")
    
    add_files("c_api/ntt_c_api.cu")
    add_headerfiles("c_api/*.h")
    add_headerfiles("../../ntt/src/*.cuh")
    add_cugencodes("native")

    set_targetdir("../../lib")