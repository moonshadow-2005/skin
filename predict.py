import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams
from matplotlib import font_manager as fm

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

    rcParams['font.family'] = 'sans-serif'
    if chosen is not None:
        rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        rcParams['font.sans-serif'] = ['DejaVu Sans']
        print('警告: 未检测到中文字体，图中中文可能显示为方块。建议安装 Noto CJK 或文泉驿字体。')

    rcParams['axes.unicode_minus'] = False


_configure_chinese_font()


def imread_unicode(image_path, flags=cv2.IMREAD_COLOR):
    """兼容Windows中文路径的读图函数。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def get_dominant_angle(angles, bin_size=5):
    """修复版本：处理角度折叠问题的主方向计算"""
    degrees = np.degrees(angles) % 180
    bins = np.arange(0, 180 + bin_size, bin_size)
    hist, bin_edges = np.histogram(degrees, bins=bins)
    
    # 检查是否存在边界折叠问题
    # 如果首尾bin都有较多数据，可能存在折叠
    first_bins_sum = np.sum(hist[:2])  # 前两个bin
    last_bins_sum = np.sum(hist[-2:])  # 后两个bin
    total_count = np.sum(hist)
    
    # 如果首尾bin占总数的比例较大，可能存在折叠
    if (first_bins_sum + last_bins_sum) / total_count > 0.3:
        # 尝试将角度转换到不同的范围进行计算
        # 方法1：转换到(-90, 90)范围
        degrees_shifted = degrees.copy()
        degrees_shifted[degrees_shifted > 90] -= 180
        
        # 重新计算直方图
        bins_shifted = np.arange(-90, 90 + bin_size, bin_size)
        hist_shifted, bin_edges_shifted = np.histogram(degrees_shifted, bins=bins_shifted)
        
        # 比较原始和shifted版本的峰值尖锐度
        max_hist_original = np.max(hist)
        max_hist_shifted = np.max(hist_shifted)
        
        # 如果shifted版本的峰值更明显，使用shifted版本
        if max_hist_shifted > max_hist_original:
            hist = hist_shifted
            bin_edges = bin_edges_shifted
            degrees = degrees_shifted
    
    # 找到峰值bin
    max_bin = np.argmax(hist)
    lower, upper = bin_edges[max_bin], bin_edges[max_bin+1]
    mask = (degrees >= lower) & (degrees < upper)
    dominant_angles = degrees[mask]
    
    if len(dominant_angles) == 0:
        # 如果没有找到主导角度，使用向量平均法
        return calculate_circular_mean_degrees(np.degrees(angles) % 180)
    
    # 使用向量平均法计算bin内的平均角度
    mean_cos = np.mean(np.cos(np.radians(dominant_angles)))
    mean_sin = np.mean(np.sin(np.radians(dominant_angles)))
    result_angle = np.degrees(np.arctan2(mean_sin, mean_cos))
    
    # 确保结果在0-180范围内
    return result_angle % 180

def calculate_circular_mean_degrees(degrees):
    """计算角度的圆形均值，处理0-180度范围的角度折叠问题"""
    # 将角度转换为弧度
    radians = np.radians(degrees)
    
    # 由于我们处理的是0-180度的方向角度（无方向性），
    # 需要将其转换为0-360度的有向角度来计算
    # 方法：将角度乘以2来消除180度的周期性
    doubled_radians = 2 * radians
    
    # 计算向量平均
    mean_cos = np.mean(np.cos(doubled_radians))
    mean_sin = np.mean(np.sin(doubled_radians))
    
    # 计算平均角度
    mean_doubled_angle = np.arctan2(mean_sin, mean_cos)
    
    # 将结果除以2并转换为度
    mean_angle = np.degrees(mean_doubled_angle / 2)
    
    # 确保结果在0-180范围内
    return mean_angle % 180

def calculate_direction_consistency(angles):
    """改进的方向一致性计算 - 修复角度折叠问题"""
    if len(angles) == 0:
        return {
            'vector_method': 0,
            'std_method': 0, 
            'deviation_method': 0,
            'std_degrees': 0
        }
    
    # 将角度转换到0-180度范围
    degrees = np.degrees(angles) % 180
    
    # 方法1: 修复版本的标准差计算
    # 使用圆形统计方法计算标准差
    circular_mean = calculate_circular_mean_degrees(degrees)
    
    # 计算角度差，考虑180度的周期性
    angle_diffs = np.abs(degrees - circular_mean)
    # 处理跨越边界的情况
    angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
    
    # 圆形标准差
    circular_std = np.sqrt(np.mean(angle_diffs**2))
    consistency_std = np.exp(-circular_std / 20)
    
    # 方法2: 向量平均法（修复版本）
    # 使用doubled angle方法来处理0-180度的周期性
    doubled_angles = 2 * np.radians(degrees)
    mean_cos = np.mean(np.cos(doubled_angles))
    mean_sin = np.mean(np.sin(doubled_angles))
    consistency_vector = np.sqrt(mean_cos**2 + mean_sin**2)
    
    # 方法3: 基于角度差的一致性（修复版本）
    mean_angle_rad = np.radians(circular_mean)
    angles_rad = np.radians(degrees)
    
    # 计算角度差，处理周期性
    angle_diffs_rad = np.abs(angles_rad - mean_angle_rad)
    angle_diffs_rad = np.minimum(angle_diffs_rad, np.pi - angle_diffs_rad)
    
    mean_deviation = np.mean(angle_diffs_rad)
    consistency_deviation = np.exp(-mean_deviation * 4)
    
    return {
        'vector_method': consistency_vector,
        'std_method': consistency_std, 
        'deviation_method': consistency_deviation,
        'std_degrees': circular_std,
        'circular_mean': circular_mean  # 添加圆形均值
    }

def calculate_outward_direction(angles, sector_center_angle):
    """
    将0-180度的无向纹理角度转换为有向的向外方向
    
    Args:
        angles: 扇区内的纹理角度数组 (0-180度)
        sector_center_angle: 扇区中心的径向角度 (0-360度)
    
    Returns:
        向外的主方向角度 (0-360度)
    """
    if len(angles) == 0:
        return sector_center_angle  # 如果没有纹理，直接返回径向方向
    
    # 计算0-180度范围内的主方向
    from predict import calculate_circular_mean_degrees
    main_direction_180 = calculate_circular_mean_degrees(np.degrees(angles) % 180)
    
    # 将0-180度的角度转换为两个可能的0-360度方向
    direction1 = main_direction_180
    direction2 = (main_direction_180 + 180) % 360
    
    # 计算扇区的径向向外方向 (从中心指向扇区中心)
    outward_direction = sector_center_angle
    
    # 计算两个可能方向与径向向外方向的角度差
    def angle_difference(a1, a2):
        """计算两个角度之间的最小差值 (0-180度)"""
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)
    
    diff1 = angle_difference(direction1, outward_direction)
    diff2 = angle_difference(direction2, outward_direction)
    
    # 选择更接近径向向外方向的角度
    if diff1 < diff2:
        return direction1
    else:
        return direction2


def quantize_to_8_directions(angle_deg):
    """Snap angle to nearest one of 8 directions: 0,45,...,315."""
    return float((int(np.round(angle_deg / 45.0)) * 45) % 360)


def get_axis_aligned_bbox_center(binary_mask):
    """Return center of the minimum axis-aligned rectangle that covers mask pixels."""
    ys, xs = np.where(binary_mask)
    if ys.size == 0:
        return None, None, None

    x1 = int(np.min(xs))
    x2 = int(np.max(xs))
    y1 = int(np.min(ys))
    y2 = int(np.max(ys))
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    return (cx, cy), (x1, y1, x2, y2), (xs, ys)

def plot_radial_sectors(img, orientations, mask, output_dir, step=20):
    h, w = img.shape
    cx, cy = w // 2, h // 2

    # 生成每个像素的极角（范围 0 ~ 2π）
    Y_all, X_all = np.indices((h, w))
    angles_from_center = np.arctan2(Y_all - cy, X_all - cx) % (2 * np.pi)

    # 分8段，每段45度（π/4）
    num_sectors = 8
    sector_masks = []
    sector_densities = []  # 存储每个扇区的密集程度
    sector_info = []  # 存储每个扇区的详细信息
    
    # 修改：直接基于像素值识别纹理像素
    # 将图像归一化到0-1范围
    img_normalized = img.astype(np.float32) / 255.0
    # 像素值大于0.4的认为是纹理像素
    texture_threshold = 0.4
    texture_mask = (img_normalized > texture_threshold) & (mask > 0)
    
    print(f"纹理识别方法: 像素值阈值法")
    print(f"纹理像素阈值: {texture_threshold} (归一化值)")
    print(f"总纹理像素数: {np.sum(texture_mask)}")
    print(f"总有效像素数: {np.sum(mask > 0)}")
    print(f"全图纹理密度比例: {np.sum(texture_mask) / np.sum(mask > 0):.3f}")
    
    for i in range(num_sectors):
        start_angle = i * (np.pi / 4)
        end_angle = (i + 1) * (np.pi / 4)
        sector_mask = (angles_from_center >= start_angle) & (angles_from_center < end_angle) & (mask > 0)
        sector_masks.append(sector_mask)
        
        # 计算扇区中心的径向角度
        sector_center_angle = (start_angle + end_angle) / 2
        sector_center_angle_deg = np.degrees(sector_center_angle)
        
        # 计算该扇区的密集程度指标
        if np.any(sector_mask):
            # 1. 扇区内有效像素总数
            total_valid_pixels = np.sum(sector_mask)
            
            # 2. 扇区内纹理像素数量
            texture_pixels_in_sector = np.sum(texture_mask & sector_mask)
            
            # 3. 纹理密度比例
            if total_valid_pixels > 0:
                texture_density_ratio = texture_pixels_in_sector / total_valid_pixels
            else:
                texture_density_ratio = 0
            
            # 4. 方向一致性（该区域纹理方向的集中程度）
            angles = orientations[sector_mask]
            angles = angles[~np.isnan(angles)]
            if len(angles) > 0:
                consistency_result = calculate_direction_consistency(angles)
                direction_consistency = consistency_result['std_method']  # 使用更敏感的方法
                std_degrees = consistency_result['std_degrees']
                vector_consistency = consistency_result['vector_method']
                
                # 新增：计算向外的主方向 (0-360度)
                outward_main_direction = calculate_outward_direction(angles, sector_center_angle_deg)
                
                # 同时保留原始的0-180度主方向用于统计
                circular_mean_180 = consistency_result.get('circular_mean', 0)
                if circular_mean_180 == 0 and len(angles) > 0:
                    circular_mean_180 = calculate_circular_mean_degrees(np.degrees(angles) % 180)
            else:
                direction_consistency = 0
                std_degrees = 0
                vector_consistency = 0
                circular_mean_180 = 0
                outward_main_direction = sector_center_angle_deg
            
            # 综合密集程度评分（纹理密度比例70% + 方向一致性30%）
            density_score = texture_density_ratio * 0.7 + direction_consistency * 0.3
            
            sector_densities.append(density_score)
            sector_info.append({
                'index': i,
                'angle_range': f"{i * 45}°~{(i + 1) * 45}°",
                'total_valid_pixels': total_valid_pixels,
                'texture_pixels': texture_pixels_in_sector,
                'texture_density_ratio': texture_density_ratio,
                'direction_consistency': direction_consistency,
                'vector_consistency': vector_consistency,  # 保留原始方法
                'std_degrees': std_degrees,  # 添加标准差信息
                'density_score': density_score,
                'main_direction_deg': circular_mean_180,  # 保留0-180度用于统计
                'outward_direction_deg': outward_main_direction,  # 新增：0-360度向外方向
                'sector_center_angle': sector_center_angle_deg  # 扇区中心径向角度
            })
        else:
            sector_densities.append(0)
            sector_info.append({
                'index': i,
                'angle_range': f"{i * 45}°~{(i + 1) * 45}°",
                'total_valid_pixels': 0,
                'texture_pixels': 0,
                'texture_density_ratio': 0,
                'direction_consistency': 0,
                'vector_consistency': 0,
                'std_degrees': 0,
                'density_score': 0,
                'main_direction_deg': 0,
                'outward_direction_deg': i * 45 + 22.5,  # 默认为扇区中心方向
                'sector_center_angle': i * 45 + 22.5
            })

    # 找出密集程度最高的扇区
    max_density_index = np.argmax(sector_densities)
    max_density_info = sector_info[max_density_index]
    
    print(f"\n=== 扇区纹理密集程度分析 ===")
    print(f"纹理识别方法: 像素值阈值法 (阈值={texture_threshold})")
    print("-" * 80)
    for info in sector_info:
        print(f"扇区{info['index']} ({info['angle_range']}): "
              f"有效像素={info['total_valid_pixels']}, "
              f"纹理像素={info['texture_pixels']}, "
              f"纹理密度={info['texture_density_ratio']:.3f}, "
              f"方向一致性={info['direction_consistency']:.3f}, "
              f"主方向={info['main_direction_deg']:.1f}°, "
              f"综合评分={info['density_score']:.3f}")
    
    print(f"\n最密集扇区: 扇区{max_density_index} ({max_density_info['angle_range']})")
    print(f"纹理密度比例: {max_density_info['texture_density_ratio']:.3f}")
    print(f"方向一致性: {max_density_info['direction_consistency']:.3f}")
    print(f"综合密集程度评分: {max_density_info['density_score']:.3f}")

    # 可视化每个扇区的方向箭头 - 使用向外方向
    plt.figure(figsize=(20, 12), dpi=200)
    for i, smask in enumerate(sector_masks):
        plt.subplot(2, 4, i + 1)
        title_color = 'red' if i == max_density_index else 'black'
        title_text = f"扇区{i}：{i * 45}°~{(i + 1) * 45}°"
        if i == max_density_index:
            title_text += "\n★最密集★"
        plt.title(title_text, color=title_color, fontweight='bold' if i == max_density_index else 'normal', fontsize=14)
        plt.imshow(img, cmap='gray')

        # 高亮当前扇形区域
        overlay_color = [255, 0, 0] if i == max_density_index else [255, 255, 0]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[smask] = overlay_color
        plt.imshow(overlay, alpha=0.4 if i == max_density_index else 0.3)
        
        # 采样该区域中的方向箭头 - 使用原始局部方向
        Y, X = np.where(smask[::step, ::step])
        Y = Y * step
        X = X * step
        local_orient = orientations[Y, X]
        U = np.cos(local_orient) * step / 2
        V = np.sin(local_orient) * step / 2
        arrow_color = 'red' if i == max_density_index else 'cyan'
        plt.quiver(X, Y, U, V, color=arrow_color, scale=1, scale_units='xy', angles='xy', width=0.003)

        # 修改：使用向外的主方向绘制主方向箭头
        outward_angle_deg = sector_info[i]['outward_direction_deg']
        outward_angle_rad = np.radians(outward_angle_deg)
        arrow_len = max(h, w) // 6
        dx = np.cos(outward_angle_rad) * arrow_len
        dy = np.sin(outward_angle_rad) * arrow_len
        main_arrow_color = 'red' if i == max_density_index else 'yellow'
        plt.quiver(cx, cy, dx, dy, color=main_arrow_color, angles='xy', scale_units='xy', scale=1,
                   width=0.012, headwidth=6, headlength=8)
        plt.text(cx + 20, cy + 20,
                 f"向外方向:{outward_angle_deg:.1f}°\n密度:{sector_densities[i]:.3f}",
                 color=main_arrow_color, fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.7))

        plt.axis('off')

    plt.tight_layout()
    
    # 修改：添加编号到文件名
    num = getattr(plot_radial_sectors, 'current_num', '001')
    output_path = os.path.join(output_dir, f'spatial_sector_directions_{num}.png')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)  # 进一步提高保存时的DPI
    plt.close()
    print(f"高分辨率图像空间扇区方向图已保存至：{output_path}")
    
    # === 为每个扇区生成专门的分析图像 ===
    # 从全局变量获取编号
    
    for i, sector_mask in enumerate(sector_masks):
        generate_sector_analysis_image(img, orientations, sector_mask, 
                                     sector_info[i], output_dir, step, texture_mask, i == max_density_index, num)
    
    # === 保存扇区信息供报告模块使用 ===
    from report import save_sector_info
    save_sector_info(sector_info, num)
    
    densest_mask = sector_masks[max_density_index]
    densest_center, densest_bbox, _ = get_axis_aligned_bbox_center(densest_mask)
    return max_density_index, max_density_info, densest_mask, densest_center, densest_bbox

def generate_sector_analysis_image(img, orientations, sector_mask, sector_info, output_dir, step=10, texture_mask=None, is_densest=False, num='001'):
    """为单个扇区生成专门的分析图像 - 直接保存到report目录"""
    h, w = img.shape
    cx, cy = w // 2, h // 2
    sector_index = sector_info['index']
    
    # 创建report目录结构
    report_dir = 'report'
    pic_dir = os.path.join(report_dir, f'{num}_pic')
    os.makedirs(pic_dir, exist_ok=True)
    
    # 创建高分辨率图像 - 修改为2x2布局
    plt.figure(figsize=(12, 8))
    
    # 设置总标题
    fig_title = f"扇区{sector_index} ({sector_info['angle_range']}) 详细分析"
    if is_densest:
        fig_title += " ★最密集扇区★"
    plt.suptitle(fig_title, fontsize=16, color='red' if is_densest else 'black', fontweight='bold')
    
    # 子图1: 扇区位置图
    plt.subplot(2, 2, 1)
    plt.title("扇区位置", fontsize=12)
    plt.imshow(img, cmap='gray')
    
    # 高亮扇区
    highlight = np.zeros((h, w, 3), dtype=np.uint8)
    highlight[sector_mask] = [255, 0, 0] if is_densest else [255, 255, 0]
    plt.imshow(highlight, alpha=0.5)
    plt.axis('off')
    
    # 子图2: 方向箭头详细图
    plt.subplot(2, 2, 2)
    plt.title(f"纹理方向详图\n方向一致性: {sector_info['direction_consistency']:.3f}", fontsize=12)
    
    # 只显示当前扇区
    masked_img = img.copy()
    masked_img[~sector_mask] = 0
    plt.imshow(masked_img, cmap='gray')
    
    # 绘制该区域的方向箭头
    Y, X = np.where(sector_mask[::step, ::step])
    Y = Y * step
    X = X * step
    if len(Y) > 0:
        local_orient = orientations[Y, X]
        U = np.cos(local_orient) * step / 2
        V = np.sin(local_orient) * step / 2
        plt.quiver(X, Y, U, V, color='cyan', scale=1, scale_units='xy', 
                   angles='xy', width=0.003, alpha=0.8)
    
    # 计算并显示该区域的主方向
    angles = orientations[sector_mask]
    angles = angles[~np.isnan(angles)]
    if len(angles) > 0:
        # 修复版本：使用圆形均值计算主方向
        mean_deg = calculate_circular_mean_degrees(np.degrees(angles) % 180)
        mean_angle = np.radians(mean_deg)
        arrow_len = max(h, w) // 4
        dx = np.cos(mean_angle) * arrow_len
        dy = np.sin(mean_angle) * arrow_len
        plt.quiver(cx, cy, dx, dy, color='red', angles='xy', scale_units='xy', scale=1,
                   width=0.015, headwidth=8, headlength=12)
        plt.text(cx + 30, cy + 30,
                 f"主方向: {mean_deg:.1f}°",
                 color='red', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='yellow', alpha=0.8))
    plt.axis('off')
    
    # 子图3: 统计信息
    plt.subplot(2, 2, 3)
    plt.title("详细统计", fontsize=12)
    
    stats_text = f"""
扇区编号: {sector_info['index']}
角度范围: {sector_info['angle_range']}

有效像素总数: {sector_info['total_valid_pixels']}
纹理像素数量: {sector_info['texture_pixels']}
纹理密度比例: {sector_info['texture_density_ratio']:.3f}

方向一致性: {sector_info['direction_consistency']:.3f}
主方向角度: {sector_info['main_direction_deg']:.1f}°
综合密集度评分: {sector_info['density_score']:.3f}

评分权重:
• 纹理密度比例: 70%
• 方向一致性: 30%

纹理识别方法: 像素值阈值法
纹理阈值: 0.4 (归一化值)
    """
    
    plt.text(0.05, 0.5, stats_text, fontsize=9, 
             verticalalignment='center',
             bbox=dict(facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    
    # 子图4: 方向分布直方图
    plt.subplot(2, 2, 4)
    plt.title("方向角度分布", fontsize=12)
    
    if len(angles) > 0:
        degrees = np.degrees(angles) % 180
        plt.hist(degrees, bins=15, alpha=0.7, color='orange', edgecolor='black')
        # 修复版本：使用圆形均值计算主方向
        mean_angle_deg = calculate_circular_mean_degrees(degrees)
        plt.axvline(mean_angle_deg, color='red', linestyle='--', 
                   linewidth=2, label=f'主方向: {mean_angle_deg:.1f}°')
        plt.xlabel('角度 (度)')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 修复版本：使用圆形统计计算标准差
        circular_mean = calculate_circular_mean_degrees(degrees)
        angle_diffs = np.abs(degrees - circular_mean)
        angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
        circular_std = np.sqrt(np.mean(angle_diffs**2))
        
        # 添加统计信息
        plt.text(0.02, 0.98, f'样本数: {len(degrees)}\n圆形标准差: {circular_std:.1f}°', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    else:
        plt.text(0.5, 0.5, '无有效数据', ha='center', va='center', 
                fontsize=14, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # 直接保存到report目录的pic文件夹
    output_path = os.path.join(pic_dir, f'sector_{sector_index:02d}_analysis.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"扇区{sector_index}专门分析图已直接保存至: {output_path}")

def analyze_texture_orientation(image_path):
    output_dir = 'predict_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取编号并设置为全局变量
    parts = os.path.basename(image_path).split('_')
    num = parts[-1].split('.')[0]
    plot_radial_sectors.current_num = num  # 修改：设置到plot_radial_sectors函数
    
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    _, mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    dx = np.zeros_like(img, dtype=np.float32)
    dy = np.zeros_like(img, dtype=np.float32)
    dx[mask > 0] = cv2.Scharr(img, cv2.CV_32F, 1, 0)[mask > 0]
    dy[mask > 0] = cv2.Scharr(img, cv2.CV_32F, 0, 1)[mask > 0]

    J11 = dx**2
    J22 = dy**2
    J12 = dx * dy

    sigma = max(3, int(min(img.shape)/100))
    ksize = (6*sigma + 1, 6*sigma + 1)
    J11 = cv2.GaussianBlur(J11, ksize, sigma)
    J22 = cv2.GaussianBlur(J22, ksize, sigma)
    J12 = cv2.GaussianBlur(J12, ksize, sigma)

    orientations = np.full(img.shape, np.nan, dtype=np.float32)
    rows, cols = np.where(mask > 0)
    for i, j in zip(rows, cols):
        ST = np.array([[J11[i,j], J12[i,j]], [J12[i,j], J22[i,j]]])
        eigenvalues, eigenvectors = np.linalg.eigh(ST)
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if main_direction[0] < 0:
            main_direction = -main_direction
        orientations[i,j] = np.arctan2(main_direction[1], main_direction[0]) + np.pi/2

    valid_orientations = orientations[~np.isnan(orientations)]
    if len(valid_orientations) == 0:
        print("警告: 未检测到有效纹理方向")
        return
    
    dominant_angle = get_dominant_angle(valid_orientations, bin_size=5)
    mean_angle = np.radians(dominant_angle)

    # === 整图方向可视化 ===
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("原始纹理图")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("纹理方向图")
    step = 20
    valid_points = mask[::step, ::step] > 0
    Y, X = np.where(valid_points)
    Y = Y * step
    X = X * step
    U = np.cos(orientations[Y, X]) * step / 2
    V = np.sin(orientations[Y, X]) * step / 2
    plt.quiver(X, Y, U, V, color='red', scale=1, scale_units='xy', angles='xy', width=0.003)

    center_x, center_y = img.shape[1]//2, img.shape[0]//2
    arrow_length = max(img.shape)//4
    plt.quiver(center_x, center_y, 
              np.cos(mean_angle)*arrow_length, 
              np.sin(mean_angle)*arrow_length, 
              color='yellow', scale=1, scale_units='xy', 
              angles='xy', width=0.01, headwidth=8, headlength=10)
    plt.text(center_x + 20, center_y + 20, 
             f"主方向: {dominant_angle:.1f}°", 
             color='yellow', fontsize=12, 
             bbox=dict(facecolor='black', alpha=0.5))
    plt.imshow(img, cmap='gray', alpha=0.3)

    base_name = parts[-1].split('.')[0]
    output_path = os.path.join(output_dir, f"orientation_texture_line_{base_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"整图纹理方向图已保存至: {output_path}")

    # === 空间扇形区域方向分析 + 密集程度分析 ===
    max_density_index, max_density_info, densest_mask, densest_center, densest_bbox = plot_radial_sectors(
        img, orientations, mask, output_dir
    )

    # === 生成单独方向图（主箭头来自最密集扇区）===
    plt.figure(figsize=(8, 8), facecolor='none')  # 透明背景
    plt.axis('off')

    # 绘制红色纹理线条
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * [1, 0, 0], alpha=0.3)  # 红色纹理

    # 绘制蓝色局部方向箭头
    plt.quiver(
        X,
        Y,
        U,
        V,
        color='blue',
        scale=1,
        scale_units='xy',
        angles='xy',
        width=0.002,
        headwidth=4,
    )

    # 使用最密集扇区主方向（连续角度，不做8方向量化）。
    arrow_direction_deg = float(max_density_info.get('main_direction_deg', max_density_info.get('outward_direction_deg', 0.0)))
    arrow_direction_rad = np.radians(arrow_direction_deg)

    if densest_center is None:
        arrow_start_x, arrow_start_y = center_x, center_y
    else:
        arrow_start_x, arrow_start_y = densest_center

    # 箭头长度按图像大小缩放。
    arrow_length = max(30, int(round(min(img.shape[:2]) * 0.18)))
    dx_main = np.cos(arrow_direction_rad) * arrow_length
    dy_main = np.sin(arrow_direction_rad) * arrow_length

    # 绘制最密集扇区包裹最小水平矩形
    if densest_bbox is not None:
        x1, y1, x2, y2 = densest_bbox
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='lime', linewidth=2)

    # 从最密集扇区包裹矩形中心绘制主箭头
    plt.quiver(
        arrow_start_x,
        arrow_start_y,
        dx_main,
        dy_main,
        color='yellow',
        scale=1,
        scale_units='xy',
        angles='xy',
        width=0.008,
        headwidth=8,
        headlength=10,
    )
    plt.text(
        arrow_start_x + 20,
        arrow_start_y + 20,
        f"最密集扇区: {max_density_index}\n主方向: {arrow_direction_deg:.1f}°",
        color='yellow',
        fontsize=12,
        bbox=dict(facecolor='black', alpha=0.5),
    )

    # 同时保存一份orientation_only文件
    orientation_only_path2 = os.path.join(output_dir, f"orientation_only_texture_line_{base_name}.png")
    plt.savefig(
        orientation_only_path2,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True,
    )

    print(f"单独方向图已同时保存至: {orientation_only_path2}")

    plt.close()
    
    print(f"\n=== 分析完成 ===")
    print(f"最密集的纹理方向位于扇区{max_density_index} ({max_density_info['angle_range']})")
    print(f"该区域的综合密集程度评分为: {max_density_info['density_score']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供图片编号")
        print("用法: python predict.py <数字>")
        sys.exit(1)
    
    num = sys.argv[1].split('.')[0]
    input_path = f"skin_output/only_texture_line_{num}.png"
    
    if not os.path.exists(input_path):
        input_path = f"skin_output/only_texture_line_{num}.jpg"
    
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 {input_path} 不存在，请确认：")
        print("1. 已运行 skin.py 生成纹理图")
        print("2. 输入编号与 skin.py 使用一致")
        sys.exit(1)
    
    analyze_texture_orientation(input_path)
