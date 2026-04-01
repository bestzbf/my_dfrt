#!/usr/bin/env python3
"""修复旧checkpoint中的head_conf权重"""

import torch
import argparse

def fix_checkpoint(input_path, output_path):
    """
    加载旧checkpoint，重新初始化head_conf层，保存新checkpoint
    """
    print(f"加载checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # 找到所有head_conf相关的键
    model_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    conf_keys = [k for k in model_dict.keys() if 'head_conf' in k]

    print(f"\n找到 {len(conf_keys)} 个head_conf相关的参数:")
    for key in conf_keys:
        old_shape = model_dict[key].shape
        print(f"  {key}: {old_shape}")

        # 重新初始化（使用Xavier初始化）
        if 'weight' in key:
            torch.nn.init.xavier_uniform_(model_dict[key])
        elif 'bias' in key:
            torch.nn.init.zeros_(model_dict[key])

    print(f"\n保存修复后的checkpoint: {output_path}")
    torch.save(checkpoint, output_path)
    print("完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入checkpoint路径")
    parser.add_argument("--output", required=True, help="输出checkpoint路径")
    args = parser.parse_args()

    fix_checkpoint(args.input, args.output)
