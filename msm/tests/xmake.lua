target("test-bn254")
    if is_mode("debug") then
        set_symbols("debug")
    end
    set_languages(("c++20"))
    add_files("bn254.cu")
    add_cugencodes("native")
    add_packages("doctest")

-- target("test-msm")
--     set_languages(("c++20"))
--     add_defines("CURVE_BN254")
--     add_files("msm.cu")
--     add_files("../src/fast_compile/msm_bn254_16_16_f.cu")
--     add_cugencodes("native")

option("m")
    set_default("bn254")
    set_showmenu(true)
    set_description("Curve for MSM configuration")
option_end()

option("k")
    set_default(20)
    set_showmenu(true)
    set_description("K parameter for MSM configuration")
option_end()

option("n")
    set_default(1)
    set_showmenu(true)
    set_description("N parameter for MSM configuration")
option_end()

-- 编译命令：xmake f --m="bn254" --k=20 --n=1 -c && xmake build test-msm
-- 编译命令：xmake f --m="bls12381" --k=20 --n=1 -c && xmake build test-msm
-- 编译命令：xmake f --m="mnt4753" --k=20 --n=1 -c && xmake build test-msm
target("test-msm")
    set_kind("binary")
    set_languages(("c++20"))
    add_cuflags("-arch=sm_89", "-Xcompiler -fPIC")
    add_cugencodes("sm_89")
    add_links("cudart", "cuda")
    add_linkdirs("/usr/local/cuda/lib64")
    add_packages("cuda")
    add_ldflags("-Wl,-rpath,/usr/local/cuda/lib64")
    on_config(function(target)
        -- 正确获取配置参数
        local m = get_config("m")
        local k = get_config("k")
        local n = get_config("n")
        local p, l
        if m == "bn254" then
            p = 255
            l = 255
        elseif m == "bls12381" then
            p = 381
            l = 256
        elseif m == "mnt4753" then
            p = 768
            l = 768
        else
            raise("Unsupported m value: " .. tostring(m))
        end
        
        -- 运行python代码获取输出
        local script_path = path.join(os.scriptdir(), "/../cost_model.py")
        local cmd = string.format("python %s %d %d %d %d", script_path, k, n, p, l)
            
        local output = os.iorun(cmd):trim()
        
        local lines = output:split("\n")
        assert(#lines == 4, "Expected 4 lines of output from python script")
        
        -- 根据输出配置参数
        target:add("defines",
            "CURVE_" .. m:upper(), 
            "ALPHA=" .. lines[1],
            "WINDOW_S=" .. lines[2],
            "BATCH_SIZE=" .. n,
            "PARTS=" .. lines[3],
            "BATCH_PER_RUN=" .. lines[4])

        -- 添加相关文件
        local filename = string.format("msm_%s_%s_%s_f.cu", m, lines[2], lines[1])
        local dynamic_file = path.join(os.scriptdir(), "../src/fast_compile", filename)
            
        if not os.isfile(dynamic_file) then
            raise("文件不存在: " .. dynamic_file)
        end

        target:set("files", {})
        target:add("files", path.join(os.scriptdir(), "msm.cu"))
        target:add("files", dynamic_file)
    end)
    add_cugencodes("native")

target("test-msm_bn254")
    set_languages(("c++20"))
    add_defines("CURVE_BN254")
    add_files("msm.cu")
    add_files("../src/fast_compile/msm_bn254_16_16_f.cu")
    add_cugencodes("native")

target("test-msm_bls12381")
    set_languages(("c++20"))
    add_defines("CURVE_BLS12381")
    add_files("msm.cu")
    add_files("../src/fast_compile/msm_bls12381_16_16_f.cu")
    add_cugencodes("native")

target("test-msm_mnt4753")
    set_languages(("c++20"))
    add_defines("CURVE_MNT4753")
    add_defines("ALPHA=64")
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
