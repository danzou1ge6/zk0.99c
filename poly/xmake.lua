target("test-poly")
    set_languages(("c++20"))
    add_files("tests/simple_test.cu")
    add_packages("doctest")
    add_headerfiles("src/*.cuh")

target("test-poly-eval")
    add_packages("doctest")
    add_files("tests/test_eval.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")

target("test-poly-kate")
    add_packages("doctest")
    add_files("tests/test_kate.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")