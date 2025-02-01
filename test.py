import subprocess

# 定义参数范围
# a_values = range(8, 25)  # a 从 8 到 24
# a_values = range(16, 17)  # a 从 8 到 16
a_values = [8, 12, 15, 16, 17, 20, 24]
b_values = [2, 4, 8, 16, 32]  # b 的取值
# c_values = [20, 22, 24, 26]  # c 的取值
c_values = [26]  # c 的取值

# 创建 .sh 文件
sh_file = "run.sh"
with open(sh_file, "w") as f:
    f.write("#!/bin/bash\n\n")  # 添加 shebang 行，确保文件可执行

    # 遍历所有可能的 (a, b, c) 组合，写入命令到 .sh 文件
    for c in c_values:
        # 输入文件路径
        input_file = f"/home/qianyu/zk0.99c/msm/tests/msm{c}.input"
        for a in a_values:
            for b in b_values:
                # 生成 test_msm_a_b 命令
                test_name = f"test_msm_{a}_{b}"
                command = f"xmake run {test_name} {input_file}\n"
                f.write(command)

# 使 .sh 文件可执行
subprocess.run(f"chmod +x {sh_file}", shell=True)