from Unet import UNet

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2
from PIL import Image, ImageDraw

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 方法二：指定具体字体路径（如果方法一无效）
# font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑

# 以下是单张图像测试代码，应放入一个新文件如test_unet.py中

def visualize_prediction(image_path, model_path='best_unet_model.pth', output_dir='results', device='cuda'):
    """
    使用训练好的模型对单张图像进行分割预测并可视化
    
    参数:
        image_path: 输入图像路径
        model_path: 模型权重文件路径
        output_dir: 结果保存目录
        device: 运行设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型 {model_path}...")
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载完成")
    
    # 读取图像
    print(f"处理图像: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 保存原始尺寸用于后处理
    original_size = (image.shape[1], image.shape[0])
    original_image = image.copy()
    
    # 预处理图像
    image = cv2.resize(image, (256, 256))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    # 模型预测
    print("开始预测...")
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    print("预测完成")
    
    # 新增：加载真实标注
    json_path = image_path.replace('.jpg', '.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 生成真实掩码
    true_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
    img = Image.new('L', original_size, 0)
    draw = ImageDraw.Draw(img)
    
    # 绘制KELOID_BOUNDARY
    for shape in data['shapes']:
        if shape['label'] == 'KELOID_BOUNDARY':
            points = [(p[0], p[1]) for p in shape['points']]
            draw.polygon(points, fill=1)
    boundary_mask = np.array(img)
    true_mask[boundary_mask == 1] = 1
    
    # 绘制KELOID_BODY
    img = Image.new('L', original_size, 0)
    draw = ImageDraw.Draw(img)
    for shape in data['shapes']:
        if shape['label'] == 'KELOID_BODY':
            points = [(p[0], p[1]) for p in shape['points']]
            draw.polygon(points, fill=1)
    body_mask = np.array(img)
    true_mask[body_mask == 1] = 2

    # 调整真实掩码大小以匹配预测结果
    true_mask_resized = cv2.resize(true_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # 新增：计算损失
    criterion = nn.CrossEntropyLoss()
    true_mask_tensor = torch.from_numpy(true_mask_resized).long().to(device)
    output = model(image_tensor)  # 重新获取输出
    loss = criterion(output, true_mask_tensor.unsqueeze(0))
    
    # 将预测结果调整回原始大小
    pred_resized = cv2.resize(pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    
    # 创建边界掩码用于可视化
    def create_boundary_mask(pred_mask, original_size):
        # 调整回原始尺寸
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建空的三通道图像
        boundary_mask = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
        
        # 仅保留两条分割线：类别1和类别2
        colors = {
            1: [0, 255, 0],   # 受影响区域边界 - 绿色
            2: [255, 0, 0]    # 瘢痕主体边界 - 红色
        }
        thickness = 2  # 边界线宽
        
        # 为类别1和2寻找边界
        for class_id in [2, 1]:  # 先绘制主体再绘制受影响区域
            if class_id not in pred_mask:
                continue
                
            # 寻找轮廓
            contours, _ = cv2.findContours(
                (pred_mask == class_id).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 绘制边界
            cv2.drawContours(boundary_mask, contours, -1, colors[class_id], thickness)
        
        return boundary_mask

    # 创建按类别填充的彩色掩码（用于半透明叠加）
    def create_filled_color_mask(pred_mask, original_size):
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)

        # 三分类填充颜色（RGB）
        class_colors = {
            0: [70, 130, 255],   # 正常皮肤：蓝色
            1: [80, 220, 120],   # 受影响区域：绿色
            2: [255, 110, 90]    # 瘢痕主体：橙红色
        }

        for class_id, color in class_colors.items():
            color_mask[pred_mask == class_id] = color

        return color_mask
    
    # 生成边界图和彩色填充图
    boundary_mask = create_boundary_mask(pred, original_size)
    filled_color_mask = create_filled_color_mask(pred, original_size)
    
    # 创建按类别半透明叠加图
    # 不同类别使用不同透明度，确保重点区域更醒目
    alpha_map = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
    pred_resized_for_alpha = cv2.resize(pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    alpha_map[pred_resized_for_alpha == 0] = 0.12
    alpha_map[pred_resized_for_alpha == 1] = 0.28
    alpha_map[pred_resized_for_alpha == 2] = 0.40

    overlay = original_image.astype(np.float32).copy()
    for c in range(3):
        overlay[:, :, c] = (1.0 - alpha_map) * overlay[:, :, c] + alpha_map * filled_color_mask[:, :, c]

    # 叠加边界，增强分区可读性
    edge_weight = 0.75
    overlay = cv2.addWeighted(overlay.astype(np.uint8), 1.0, boundary_mask, edge_weight, 0)

    # 仅线条叠加图：把黑色背景视为透明，仅叠加彩色分割线
    overlay_lineonly = original_image.copy().astype(np.float32)
    line_alpha = 0.95
    line_pixels = np.any(boundary_mask > 0, axis=2)
    for c in range(3):
        overlay_lineonly[:, :, c][line_pixels] = (
            (1.0 - line_alpha) * overlay_lineonly[:, :, c][line_pixels]
            + line_alpha * boundary_mask[:, :, c][line_pixels]
        )
    overlay_lineonly = overlay_lineonly.astype(np.uint8)
    
    # 保存可视化结果
    output_filename = os.path.basename(image_path).split('.')[0]
    
    # 原始图像
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("原始图像")
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("分割填充图")
    plt.imshow(filled_color_mask)
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("上色叠加")
    plt.imshow(overlay)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("线条叠加")
    plt.imshow(overlay_lineonly)
    plt.axis('off')
    
    # 保存合并图像
    output_path = os.path.join(output_dir, f"{output_filename}_result.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # 单独保存各个结果
    cv2.imwrite(os.path.join(output_dir, f"{output_filename}_mask.png"), 
                cv2.cvtColor(boundary_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{output_filename}_overlay.png"), 
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{output_filename}_overlay_lineonly.png"),
                cv2.cvtColor(overlay_lineonly, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{output_filename}_filled_mask.png"),
                cv2.cvtColor(filled_color_mask, cv2.COLOR_RGB2BGR))
    
    # 输出结果分析
    unique, counts = np.unique(pred_resized, return_counts=True)
    percentages = counts / pred_resized.size * 100
    
    class_names = ["正常皮肤", "受影响区域", "瘢痕主体"]
    print("\n分割结果分析:")
    print(f"  模型损失值: {loss.item():.4f}")
    for i in range(3):
        if i in unique:
            idx = np.where(unique == i)[0][0]
            print(f"  {class_names[i]}: {percentages[idx]:.2f}% ({counts[idx]} 像素)")
        else:
            print(f"  {class_names[i]}: 0.00% (0 像素)")
    
    print(f"\n结果已保存至: {output_dir}/")
    
    return pred_resized, overlay

# 如果要作为主程序运行
if __name__ == "__main__":
    import sys
    
    # 从命令行获取数字
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
            # 构建图片路径
            data_dir = 'dataset/final_labeled'
            image_path = os.path.join(data_dir, f"{num}.jpg")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"错误: 图片 {image_path} 不存在!")
                print(f"请确保数字在1-404之间且对应的图片文件存在。")
                sys.exit(1)
        except ValueError:
            print("请输入有效的数字")
            print("用法: python test.py <数字> [模型路径]")
            sys.exit(1)
    else:
        print("请提供图片编号")
        print("用法: python test.py <数字> [模型路径]")
        sys.exit(1)
    
    # 可选模型路径
    model_path = 'best_trans_unet_model_20250614_122913.pth'  # 默认值
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 运行预测
    print(f"开始分析图片: {num}.jpg")
    visualize_prediction(image_path, model_path, device=device)