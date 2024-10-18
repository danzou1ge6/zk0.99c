target("cuda_api")
    set_kind("static")
    add_values("cuda.build.devlink", true)
    if is_mode("debug") then 
        set_symbols("debug")
    end
    
    add_files("c_api/*.cu")
    add_headerfiles("c_api/*.h")
    add_cugencodes("native")

    set_targetdir("../../lib")