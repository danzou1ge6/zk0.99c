target("test-json")
    if is_mode("debug") then 
        set_symbols("debug")
    end
    
    add_files("simple_json.cpp")

target("test-graph")
    local project_root = os.projectdir()
    set_languages("c++20")
    if is_mode("debug") then 
        set_symbols("debug")
    end
    
    add_files("test_graph.cpp")
    add_files("../src/graph.cu")
    add_files("../src/field_type.cpp")
    add_files("../src/memory.cpp")
    add_headerfiles("../src/field_type.h")
    add_headerfiles("../src/graph.cuh")
    add_headerfiles("../src/memory.h")
    add_defines("PROJECT_ROOT=\"" .. project_root .. "\"")