import cv2
import numpy as np
import sys
import torch
from Unet import UNet
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm
import os


def _configure_chinese_font() -> None:
    candidate_files = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
    ]
    for fp in candidate_files:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
            except Exception:
                pass

    fm._load_fontmanager(try_read_cache=False)

    preferred = [
        'SimHei',
        'Microsoft YaHei',
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Noto Sans CJK TC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Source Han Sans CN',
        'Arial Unicode MS',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), None)

    if chosen is not None:
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        # Fallback to DejaVu; Chinese glyphs may still be missing without CJK fonts installed.
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print('警告: 未检测到中文字体，图中中文可能显示为方块。建议安装 Noto CJK 或文泉驿字体。')

    plt.rcParams['axes.unicode_minus'] = False


_configure_chinese_font()


def imread_unicode(image_path, flags=cv2.IMREAD_COLOR):
    """兼容Windows中文路径的读图函数。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def analyze_skin_texture(image_path, model_path='best_trans_unet_model_20250614_122913.pth', device='cuda'):
    """
    皮肤纹理分析主函数
    
    参数:
        image_path: 输入图像路径
        model_path: 模型权重路径
        device: 运行设备
    """
    # 创建输出目录
    output_dir = 'skin_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 修改1：新增模型预测函数
    def predict_mask(img_path, model_path, device):
        """独立的分割预测函数"""
        # 加载模型
        model = UNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 读取并预处理图像
        raw_img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if raw_img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        img_tensor = transforms.ToTensor()(cv2.resize(image, (256,256))).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # 调整回原始尺寸
        return cv2.resize(pred.astype(np.uint8), (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_NEAREST)

    # 修改2：替换原来的visualize_prediction调用
    pred_mask = predict_mask(image_path, model_path, device)
    
    # 读取原始图像
    raw_original = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if raw_original is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    original_img = cv2.cvtColor(raw_original, cv2.COLOR_BGR2RGB)
    
    # 修改1：保留原始彩色图像
    rgb_img = original_img.copy()
    
    # 创建受影响区域掩码（标签1）
    affected_mask = (pred_mask == 1).astype(np.uint8)
    
    # 应用形态学操作去除小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_CLOSE, kernel)
    
    # 动态计算扩展量
    image_height = original_img.shape[0]
    expand_pixels = image_height // 30  # 使用整除运算
    print(f"根据图像高度{image_height}计算，扩展像素数: {expand_pixels}")

    # 生成动态核尺寸（保证奇数尺寸）
    kernel_size = 2 * expand_pixels + 1
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 执行膨胀操作
    expanded_mask = cv2.dilate(cleaned_mask, expand_kernel, iterations=1)
    
    # 提取扩展后的ROI
    affected_roi = cv2.bitwise_and(rgb_img, rgb_img, mask=expanded_mask)  # 使用扩展后的掩码
    
    # 创建Gabor滤波器组
    def build_filters():
        filters = []
        ksize = 9  # 滤波器尺寸
        for theta in np.arange(0, np.pi, np.pi / 4):  # 4个方向
            for lamda in np.arange(10, 20, 5):        # 3个波长
                params = {
                    'ksize': (ksize, ksize),
                    'sigma': 3.0,
                    'theta': theta,
                    'lambd': lamda,
                    'gamma': 0.5,
                    'psi': 0
                }
                kern = cv2.getGaborKernel(**params)
                kern /= 1.5*kern.sum()  # 归一化
                filters.append((kern, params))
        return filters
    
    # 应用Gabor滤波器组
    filters = build_filters()
    texture_map = np.zeros_like(affected_roi[:, :, 0], dtype=np.float32)
    
    # 合并多个滤波器的响应
    for kern, params in filters:
        fimg = cv2.filter2D(affected_roi, cv2.CV_32F, kern)
        texture_map += cv2.cvtColor(fimg, cv2.COLOR_RGB2GRAY)
    
    # 归一化纹理图
    texture_map = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX)
    
    # 新增：纹理线条检测
    def detect_texture_lines(texture_map):
        # 使用自适应阈值处理
        binary = cv2.adaptiveThreshold(texture_map.astype(np.uint8), 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 边缘检测
        edges = cv2.Canny(binary, 50, 150)
        
        # 形态学操作增强线条
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges

    # 生成纹理线条
    texture_lines = detect_texture_lines(texture_map)
    
    # 新增：计算收缩量
    shrink_pixels = max(1, image_height // 80)  # 至少收缩1像素
    shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (2*shrink_pixels+1, 2*shrink_pixels+1))
    
    # 应用腐蚀操作到扩展后的掩模
    shrunk_mask = cv2.erode(expanded_mask, shrink_kernel, iterations=1)
    
    # 新增：提前定义base_name
    base_name = os.path.basename(image_path).split('.')[0]
    
    # 修改单独保存纹理线条图部分（添加掩模应用）
    plt.figure(figsize=(8, 8))

    
    # 应用收缩后的掩模
    refined_lines = cv2.bitwise_and(texture_lines, texture_lines, mask=shrunk_mask)
    
    plt.imshow(refined_lines, cmap='gray')
    plt.axis('off')
    line_only_path = os.path.join(output_dir, f"only_texture_line_{base_name}.png")
    
    # 关键修改：添加保存参数
    plt.savefig(line_only_path, 
               bbox_inches='tight', 
               pad_inches=0, 
               facecolor='none', 
               transparent=True)
    
    plt.close()  # 关闭当前figure释放内存
    
    # 创建线条可视化
    plt.figure(figsize=(15, 8))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.title("原始图像")
    plt.imshow(original_img)
    plt.axis('off')
    
    # 纯纹理线条图
    plt.subplot(1, 3, 2)
    plt.title("纹理线条图")
    plt.imshow(texture_lines, cmap='gray')
    plt.axis('off')
    
    # 叠加显示（仅在受影响区域显示）
    plt.subplot(1, 3, 3)
    plt.title("线条叠加效果")
    
    # 生成彩色线条（红色）
    line_color = np.zeros_like(original_img)
    line_color[texture_lines > 0] = [255, 0, 0]  # 红色线条
    
    # 仅保留受影响区域的线条
    line_color = cv2.bitwise_and(line_color, line_color, mask=expanded_mask)
    
    # 叠加到原图
    overlay = cv2.addWeighted(original_img, 0.7, line_color, 0.3, 0)
    
    plt.imshow(overlay)
    plt.axis('off')
    
    # 修改保存路径
    output_path = os.path.join(output_dir, f"texture_line_{base_name}.png")
    
    # 在保存结果前添加掩模保存
    #mask_path = os.path.join(output_dir, f"mask_{base_name}.png")
    #cv2.imwrite(mask_path, expanded_mask * 255)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"对比图已保存至: {output_path}")
    print(f"单独线条图已保存至: {line_only_path}")

if __name__ == "__main__":
    # 构建图片路径
    num = sys.argv[1]
    data_dir = 'dataset/final_labeled'
    image_path = f"{data_dir}/{num}.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片 {image_path} 不存在!")
        sys.exit(1)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 运行分析前自动创建目录
    os.makedirs('skin_output', exist_ok=True)
    analyze_skin_texture(image_path, device=device)
