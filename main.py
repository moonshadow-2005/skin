import subprocess
import sys
import os
from pathlib import Path
import cv2
import numpy as np

def overlay_images(original_path, direction_path, output_path):
    """ 叠加原始图片和方向图 """
    # 读取原始图片
    original = cv2.imread(original_path)
    if original is None:
        print(f"错误：无法读取原始图片 {original_path}")
        return False
    
    # 读取方向图（带透明通道）
    direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)
    if direction is None:
        print(f"错误：无法读取方向图 {direction_path}")
        return False
    
    # 调整方向图尺寸与原始图片一致
    if direction.shape[:2] != original.shape[:2]:
        direction = cv2.resize(direction, (original.shape[1], original.shape[0]))
    
    # 分离方向图的颜色和alpha通道
    direction_rgb = direction[:, :, :3]
    alpha = direction[:, :, 3] / 255.0  # 转换为0-1的透明度
    
    # 创建叠加图像
    overlay = original.copy()
    for c in range(3):
        overlay[:, :, c] = (1 - alpha) * original[:, :, c] + alpha * direction_rgb[:, :, c]
    
    # 保存结果
    cv2.imwrite(output_path, overlay)
    return True

def main():
    # 在文件开头添加final_output目录创建
    final_output_dir = 'final_output'
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 检查输入参数
    if len(sys.argv) < 2:
        print("请提供图片编号")
        print("用法: python main.py <数字>")
        sys.exit(1)
    
    num = sys.argv[1]
    print(f"开始处理编号 {num} 的图片...")
    
    # 步骤1：运行skin.py生成纹理图
    print("\n=== 步骤1/4：生成皮肤纹理图 ===")
    skin_cmd = f"python skin.py {num}"
    skin_result = subprocess.run(skin_cmd, shell=True, capture_output=True, text=True)
    
    # 检查skin.py执行结果
    if skin_result.returncode != 0:
        print("skin.py执行失败：")
        print(skin_result.stderr)
        sys.exit(2)
    
    # 验证生成的纹理图
    texture_path = Path(f"skin_output/only_texture_line_{num}.png")
    if not texture_path.exists():
        print(f"错误：未找到生成的纹理图 {texture_path}")
        print("可能原因：")
        print("1. 原始图片不存在")
        print("2. skin.py执行过程中出现错误")
        sys.exit(3)
    
    # 步骤2：运行predict.py分析方向
    print("\n=== 步骤2/4：分析纹理方向 ===")
    predict_cmd = f"python predict.py {num}"
    predict_result = subprocess.run(predict_cmd, shell=True, capture_output=True, text=True)
    
    # 检查predict.py执行结果
    if predict_result.returncode != 0:
        print("predict.py执行失败：")
        print(predict_result.stderr)
        sys.exit(4)
    
    # 验证最终结果
    final_output = Path(f"predict_output/orientation_only_texture_line_{num}.png")
    if not final_output.exists():
        print(f"错误：未找到最终结果文件 {final_output}")
        sys.exit(5)
    
    # 步骤3：生成最终叠加图
    print("\n=== 步骤3/4：生成最终叠加图 ===")
    
    # 定义路径
    original_img_path = f"dataset/final_labeled/{num}.jpg"
    direction_only_path = f"predict_output/orientation_only_texture_line_{num}.png"
    final_output_path = os.path.join(final_output_dir, f"final_result_{num}.jpg")
    
    # 执行叠加
    if not overlay_images(original_img_path, direction_only_path, final_output_path):
        sys.exit(6)
    
    # 新增步骤4：生成分析报告
    print("\n=== 步骤4/4：生成分析报告 ===")
    try:
        from report import generate_report
        generate_report(num)
    except Exception as e:
        print(f"报告生成失败: {e}")
        sys.exit(7)
    
    # 更新成功信息
    print("\n=== 处理完成 ===")
    print(f"原始图片： {original_img_path}")
    print(f"方向指示图： {direction_only_path}")
    print(f"最终结果： {final_output_path}")
    print(f"分析报告： report/{num}_skin_texture_analysis_report.md")
    print(f"所有中间文件可在 predict_output/ 和 skin_output/ 中找到")

if __name__ == "__main__":
    main()
