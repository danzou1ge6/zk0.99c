target("test-json")
    if is_mode("debug") then 
        set_symbols("debug")
    end
    
    add_files("simple_json.cpp")

target("test-graph")
    set_languages("c++20")
    if is_mode("debug") then 
        set_symbols("debug")
    end
    
    add_files("test_graph.cpp")
    add_files("../src/*.cu")
    add_files("../src/*.cpp")
    add_headerfiles("../src/*.cuh")
    add_headerfiles("../src/*.h")