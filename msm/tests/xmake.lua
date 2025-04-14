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
    add_files("../src/fast_compile/msm_mnt4753_16_64_f.cu")
    add_cugencodes("native")

-- 定义所有可能的组合
-- local window_sizes = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
-- local alphas = {1, 2, 4, 8, 12, 16, 24, 32}

-- 遍历所有组合生成目标
-- for _, window_size in ipairs(window_sizes) do
--     for _, alpha in ipairs(alphas) do
--         local target_name = "test_msm_" .. window_size .. "_" .. alpha
--         target(target_name)
--             set_languages("c++20")
--             add_files("msm.cu")
--             local msm_file = string.format("../src/fast_compile/msm_bn254_%d_%d_f.cu", window_size, alpha)
--             add_files(msm_file)
--             add_cugencodes("native")

--             add_defines("WINDOW_S=" .. window_size)
--             add_defines("ALPHA=" .. alpha)
--     end
-- end
