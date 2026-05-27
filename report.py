import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
import pickle
import warnings
import sys  # 添加sys导入
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
        rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans', 'Arial Unicode MS']
    else:
        rcParams['font.sans-serif'] = ['DejaVu Sans']
        print('警告: 未检测到中文字体，图中中文可能显示为方块。建议安装 Noto CJK 或文泉驿字体。')

    rcParams['axes.unicode_minus'] = False


_configure_chinese_font()

# 添加字体回退机制
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

def generate_summary_charts(sector_info, pic_dir):
    """生成汇总对比图表"""
    densities = [info['density_score'] for info in sector_info]
    max_index = np.argmax(densities)
    
    plt.figure(figsize=(16, 10))
    
    # 密集度评分柱状图
    plt.subplot(2, 3, 1)
    sectors = [f"扇区{i}" for i in range(8)]
    scores = [info['density_score'] for info in sector_info]
    colors = ['red' if i == max_index else 'lightblue' for i in range(8)]
    bars = plt.bar(sectors, scores, color=colors)
    plt.title('各扇区密集度评分对比', fontsize=12)
    plt.ylabel('密集度评分')
    plt.xticks(rotation=45)
    
    # 标注最高分
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 纹理密度比例对比
    plt.subplot(2, 3, 2)
    density_ratios = [info['texture_density_ratio'] for info in sector_info]
    plt.bar(sectors, density_ratios, color='green', alpha=0.7)
    plt.title('各扇区纹理密度比例', fontsize=12)
    plt.ylabel('纹理密度比例')
    plt.xticks(rotation=45)
    
    # 方向一致性对比
    plt.subplot(2, 3, 3)
    consistencies = [info['direction_consistency'] for info in sector_info]
    plt.bar(sectors, consistencies, color='orange', alpha=0.7)
    plt.title('各扇区方向一致性', fontsize=12)
    plt.ylabel('方向一致性')
    plt.xticks(rotation=45)
    
    # 方向标准差对比
    plt.subplot(2, 3, 4)
    std_degrees = [info['std_degrees'] for info in sector_info]
    colors_std = ['red' if i == max_index else 'purple' for i in range(8)]
    plt.bar(sectors, std_degrees, color=colors_std, alpha=0.7)
    plt.title('各扇区方向标准差', fontsize=12)
    plt.ylabel('方向标准差 (度)')
    plt.xticks(rotation=45)
    
    # 像素数量对比
    plt.subplot(2, 3, 5)
    valid_pixels = [info['total_valid_pixels'] for info in sector_info]
    texture_pixels = [info['texture_pixels'] for info in sector_info]
    
    x = np.arange(len(sectors))
    width = 0.35
    plt.bar(x - width/2, valid_pixels, width, label='有效像素', alpha=0.7)
    plt.bar(x + width/2, texture_pixels, width, label='纹理像素', alpha=0.7)
    plt.title('各扇区像素数量对比', fontsize=12)
    plt.ylabel('像素数量')
    plt.xticks(x, sectors, rotation=45)
    plt.legend()
    
    # 综合评价文本
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
综合分析结果:

最密集扇区: 扇区{max_index}
角度范围: {sector_info[max_index]['angle_range']}
综合评分: {sector_info[max_index]['density_score']:.3f}

该扇区特点:
• 纹理密度比例: {sector_info[max_index]['texture_density_ratio']:.3f}
• 方向一致性: {sector_info[max_index]['direction_consistency']:.3f}
• 主方向角度: {sector_info[max_index]['main_direction_deg']:.1f}°
• 方向标准差: {sector_info[max_index]['std_degrees']:.1f}°

评价标准:
• 综合评分 = 纹理密度比例×70% + 方向一致性×30%
• 纹理密度比例: 纹理像素数/有效像素数
• 方向一致性: exp(-标准差/20)
    """
    
    plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    summary_chart_path = os.path.join(pic_dir, '01_summary_charts.png')
    plt.savefig(summary_chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"汇总分析图已保存至: {summary_chart_path}")

def generate_summary_report(sector_info, num):
    """生成所有扇区的汇总报告 - 使用markdown格式"""
    
    # 创建report目录结构
    report_dir = 'report'
    os.makedirs(report_dir, exist_ok=True)
    
    pic_dir = os.path.join(report_dir, f'{num}_pic')
    os.makedirs(pic_dir, exist_ok=True)
    
    # 复制概览图
    import shutil
    
    # 从predict_output复制spatial_sector_directions_{编号}.png作为概览图
    source_overview = os.path.join('predict_output', f'spatial_sector_directions_{num}.png')
    target_overview = os.path.join(pic_dir, '00_overview.png')
    
    if os.path.exists(source_overview):
        shutil.copy2(source_overview, target_overview)
        print(f"概览图已复制至: {target_overview}")
    else:
        print(f"警告: 未找到概览图源文件 {source_overview}")
    
    # 生成汇总图表
    generate_summary_charts(sector_info, pic_dir)
    
    # 创建markdown报告
    report_path = os.path.join(report_dir, f'{num}_skin_texture_analysis_report.md')
    
    # 汇总统计
    densities = [info['density_score'] for info in sector_info]
    max_index = np.argmax(densities)
    max_density_info = sector_info[max_index]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 标题和目录
        f.write(f"# 皮肤纹理方向分析报告 - 样本{num}\n\n")
        f.write("## 目录\n\n")
        f.write("- [1. 执行摘要](#1-执行摘要)\n")
        f.write("- [2. 分析方法](#2-分析方法)\n") 
        f.write("- [3. 整体分析结果](#3-整体分析结果)\n")
        f.write("- [4. 扇区详细分析](#4-扇区详细分析)\n")
        f.write("- [5. 对比分析](#5-对比分析)\n")
        
        f.write("---\n\n")
        
        # 1. 执行摘要
        f.write("## 1. 执行摘要\n\n")
        f.write(f"**分析样本**: {num}  \n")
        f.write(f"**分析日期**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**最密集纹理扇区**: 扇区{max_index} ({max_density_info['angle_range']})  \n")
        f.write(f"**综合密集度评分**: {max_density_info['density_score']:.3f}  \n")
        f.write(f"**主要纹理方向**: {max_density_info['main_direction_deg']:.1f}°  \n")
        f.write(f"**纹理密度比例**: {max_density_info['texture_density_ratio']:.3f}  \n")
        f.write(f"**方向一致性**: {max_density_info['direction_consistency']:.3f}  \n\n")
        
        # 2. 分析方法
        f.write("## 2. 分析方法\n\n")
        f.write("### 2.1 纹理像素识别\n")
        f.write("- **方法**: 像素值阈值法\n")
        f.write("- **阈值**: 0.4 (归一化像素值)\n")
        f.write("- **原理**: 像素值 > 0.4 的像素认为是纹理像素\n\n")
        
        f.write("### 2.2 空间分区策略\n")
        f.write("- **分区方法**: 径向扇形分区\n")
        f.write("- **扇区数量**: 8个\n")
        f.write("- **每扇区角度**: 45°\n")
        f.write("- **中心点**: 图像几何中心\n\n")
        
        f.write("### 2.3 评价指标\n")
        f.write("- **纹理密度比例**: 纹理像素数 / 扇区有效像素数\n")
        f.write("- **方向一致性**: 基于标准差的指数衰减评分 exp(-std/20)\n")
        f.write("- **综合评分**: 纹理密度比例×70% + 方向一致性×30%\n\n")
        
        # 3. 整体分析结果
        f.write("## 3. 整体分析结果\n\n")
        f.write("### 3.1 全局概览\n\n")
        f.write(f"![整体扇区分析概览]({num}_pic/00_overview.png)\n\n")
        f.write("*图3.1: 八个扇区的纹理方向分析概览，红色标注为最密集扇区*\n\n")
        
        f.write("### 3.2 汇总统计\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| 最高密集度评分 | {max_density_info['density_score']:.3f} |\n")
        f.write(f"| 平均密集度评分 | {np.mean(densities):.3f} |\n")
        f.write(f"| 密集度标准差 | {np.std(densities):.3f} |\n")
        f.write(f"| 扇区差异程度 | {'显著' if np.std(densities) > 0.1 else '中等' if np.std(densities) > 0.05 else '轻微'} |\n\n")
        
        # 4. 扇区详细分析
        f.write("## 4. 扇区详细分析\n\n")
        
        # 按密集度排序
        sorted_sectors = sorted(enumerate(densities), key=lambda x: x[1], reverse=True)
        
        for rank, (sector_idx, score) in enumerate(sorted_sectors):
            info = sector_info[sector_idx]
            f.write(f"### 4.{rank+1} 扇区{sector_idx} ({info['angle_range']}) ")
            if sector_idx == max_index:
                f.write("★ 最密集扇区\n\n")
            else:
                f.write(f"- 排名第{rank+1}\n\n")
            
            # 扇区图片
            f.write(f"![扇区{sector_idx}详细分析]({num}_pic/sector_{sector_idx:02d}_analysis.png)\n\n")
            f.write(f"*图4.{rank+1}: 扇区{sector_idx}的详细分析图*\n\n")
            
            # 详细数据表
            f.write("#### 详细指标\n\n")
            f.write("| 指标 | 数值 | 说明 |\n")
            f.write("|------|------|------|\n")
            f.write(f"| 有效像素总数 | {info['total_valid_pixels']} | 扇区内可分析像素数量 |\n")
            f.write(f"| 纹理像素数量 | {info['texture_pixels']} | 识别出的纹理像素数 |\n")
            f.write(f"| 纹理密度比例 | {info['texture_density_ratio']:.3f} | 纹理覆盖程度 |\n")
            f.write(f"| 方向一致性 | {info['direction_consistency']:.3f} | 方向集中程度 |\n")
            f.write(f"| 主方向角度 | {info['main_direction_deg']:.1f}° | 主要纹理方向 |\n")
            f.write(f"| 方向标准差 | {info['std_degrees']:.1f}° | 方向分散程度 |\n")
            f.write(f"| 综合密集度评分 | {info['density_score']:.3f} | 最终评分 |\n\n")
            
            # 特征描述
            f.write("#### 特征描述\n\n")
            
            # 密度等级
            if info['texture_density_ratio'] >= 0.7:
                density_level = "高密度"
            elif info['texture_density_ratio'] >= 0.5:
                density_level = "中密度"
            else:
                density_level = "低密度"
            
            # 一致性等级
            if info['direction_consistency'] >= 0.7:
                consistency_level = "高一致性"
            elif info['direction_consistency'] >= 0.4:
                consistency_level = "中等一致性"
            else:
                consistency_level = "低一致性"
            
            f.write(f"- **纹理密度**: {density_level} ({info['texture_density_ratio']:.1%})\n")
            f.write(f"- **方向特征**: {consistency_level}，标准差{info['std_degrees']:.1f}°\n")
            f.write(f"- **主导方向**: {info['main_direction_deg']:.1f}°（{'水平' if 0 <= info['main_direction_deg'] <= 30 or 150 <= info['main_direction_deg'] <= 180 else '垂直' if 60 <= info['main_direction_deg'] <= 120 else '斜向'}纹理）\n")
            
           
        
        # 5. 对比分析
        f.write("## 5. 对比分析\n\n")
        f.write("扇区对比图表\n\n")
        f.write(f"![扇区对比分析]({num}_pic/01_summary_charts.png)\n\n")
        f.write("*图5.1: 各扇区的多维度对比分析*\n\n")
        
       
        f.write("---\n")
        f.write(f"*报告生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"Markdown报告已保存至: {report_path}")
    print(f"相关图片已保存至: {pic_dir}")

def save_sector_info(sector_info, num):
    """保存扇区信息到文件，供报告模块使用"""
    output_dir = 'predict_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为pickle文件（保持数据类型）
    with open(os.path.join(output_dir, f'sector_info_{num}.pkl'), 'wb') as f:
        pickle.dump(sector_info, f)
    
    # 同时保存为JSON文件（便于查看）
    json_sector_info = []
    for info in sector_info:
        json_info = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                json_info[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_info[key] = float(value)
            else:
                json_info[key] = value
        json_sector_info.append(json_info)
    
    with open(os.path.join(output_dir, f'sector_info_{num}.json'), 'w', encoding='utf-8') as f:
        json.dump(json_sector_info, f, ensure_ascii=False, indent=2)
    
    print(f"扇区信息已保存至: predict_output/sector_info_{num}.pkl")

def load_sector_info(num):
    """从文件加载扇区信息"""
    pickle_path = os.path.join('predict_output', f'sector_info_{num}.pkl')
    
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"未找到扇区信息文件: {pickle_path}")

def generate_report(num):
    """报告生成主函数"""
    print(f"\n=== 生成分析报告 ===")
    
    try:
        # 加载扇区信息
        sector_info = load_sector_info(num)
        
        # 生成完整报告
        generate_summary_report(sector_info, num)
        
        print(f"样本{num}的完整分析报告生成完毕！")
        print(f"报告文件: report/{num}_skin_texture_analysis_report.md")
        print(f"相关图片: report/{num}_pic/")
        
    except Exception as e:
        print(f"报告生成失败: {e}")
        raise 

# 添加命令行接口支持
if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("请提供图片编号")
        print("用法: python report.py <编号>")
        print("示例: python report.py 1")
        print("      python report.py 202")
        sys.exit(1)
    
    # 获取图片编号
    num = sys.argv[1].split('.')[0]  # 去掉可能的文件扩展名
    
    # 检查扇区信息文件是否存在
    sector_info_path = os.path.join('predict_output', f'sector_info_{num}.pkl')
    if not os.path.exists(sector_info_path):
        print(f"错误：未找到扇区信息文件 {sector_info_path}")
        print("请确认：")
        print("1. 已运行过 predict.py 生成分析数据")
        print("2. 输入编号与之前分析时使用的编号一致")
        print("3. predict_output 目录存在且包含相关文件")
        sys.exit(1)
    
    print(f"开始为样本{num}生成分析报告...")
    
    # 运行报告生成
    try:
        generate_report(num)
        print(f"\n=== 报告生成完成 ===")
        print(f"您可以查看以下文件：")
        print(f"• 主报告：report/{num}_skin_texture_analysis_report.md")
        print(f"• 图片文件夹：report/{num}_pic/")
        
    except FileNotFoundError as e:
        print(f"文件未找到错误：{e}")
        print("请确保已完成前序分析步骤")
        sys.exit(1)
    except Exception as e:
        print(f"报告生成过程中发生错误：{e}")
        sys.exit(1) 