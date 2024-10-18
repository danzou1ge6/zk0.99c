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
    add_files("../src/graph.cu")
    add_headerfiles("../src/graph.cuh")