#!/usr/bin/env python3
"""
将训练模型转换为推理专用模型（减小文件大小）
"""

import argparse
import os
from dp_ai import DPAI


def convert_model(input_path: str, output_path: str = None):
    """
    将训练模型转换为推理模型
    
    Args:
        input_path: 输入模型路径（训练模型）
        output_path: 输出模型路径（推理模型，默认为input_path_inference.pkl）
    """
    if not os.path.exists(input_path):
        print(f"错误: 模型文件不存在: {input_path}")
        return
    
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_inference.pkl"
    
    print(f"加载训练模型: {input_path}")
    ai = DPAI()
    ai.load_model(input_path)
    
    # 获取文件大小
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    print(f"原始模型大小: {input_size:.2f} MB")
    
    print(f"保存推理模型: {output_path}")
    ai.save_model_for_inference(output_path)
    
    # 获取新文件大小
    output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"推理模型大小: {output_size:.2f} MB")
    
    reduction = (1 - output_size / input_size) * 100
    print(f"文件大小减少: {reduction:.1f}%")
    print(f"\n转换完成！")
    print(f"推理时可以使用: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将训练模型转换为推理模型")
    parser.add_argument("--input", type=str, default="ai_model.pkl", 
                       help="输入模型路径（训练模型）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出模型路径（推理模型，默认为input_inference.pkl）")
    
    args = parser.parse_args()
    
    convert_model(args.input, args.output)

