import os

# 参数设置
second_params = list(range(8, 25))  # 8到24
third_params = [1, 2, 4, 8, 12, 16, 24, 32]  # 给定的第三个参数

# 文件存储路径
output_dir = "."

# 遍历所有组合
for second in second_params:
    for third in third_params:
        # 构建文件名
        file_name = f"msm_bn254_{second}_{third}_f.cu"
        file_path = os.path.join(output_dir, file_name)

        # 如果文件已存在，则跳过生成
        if os.path.exists(file_path):
            print(f"File {file_name} already exists, skipping.")
            continue

        # 构建文件内容
        content = f"""#include "../msm_impl.cuh"
#include "../bn254.cuh"

namespace msm {{
    using Config = MsmConfig<255, {second}, {third}, false>;
    template class MSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
    template class MSMPrecompute<Config, bn254::Point, bn254::PointAffine>;
    template class MultiGPUMSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
}}
"""
        # 写入文件
        with open(file_path, 'w') as f:
            f.write(content)

        print(f"Generated {file_name}")