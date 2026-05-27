import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import argparse
from datetime import datetime

# 定义U-Net模型
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 下采样路径
        self.down_conv1 = DoubleConv(3, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.down_conv5 = DoubleConv(512, 1024)
        
        # 上采样路径
        self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        
        self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        # 最终输出层 (3个类别: 瘢痕主体, 受影响区域, 正常皮肤)
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        
        # 池化层
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # 下采样
        conv1 = self.down_conv1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.down_conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down_conv3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.down_conv4(x)
        x = self.maxpool(conv4)
        
        x = self.down_conv5(x)
        
        # 上采样
        x = self.up_trans1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv1(x)
        
        x = self.up_trans2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv2(x)
        
        x = self.up_trans3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv3(x)
        
        x = self.up_trans4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv4(x)
        
        x = self.out(x)
        return x

# 自定义数据集类
class KeloidDataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 256), transform=None, is_training=True, use_3x_augmentation=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_training = is_training
        self.use_3x_augmentation = use_3x_augmentation  # 新增参数
        
        # 定义数据增强变换
        if use_3x_augmentation:  # 修改条件：只要启用3倍数据增强就定义多种变换
            # 原图变换（不进行颜色抖动）
            self.transform_original = transforms.Compose([
                transforms.ToTensor()
            ])
            # 第一种颜色抖动变换
            self.transform_aug1 = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.15,     # 亮度抖动 ±15%
                    contrast=0.15,       # 对比度抖动 ±15%
                    saturation=0.15,     # 饱和度抖动 ±15%
                    hue=0.05              # 色调抖动 ±10%
                ),
                transforms.ToTensor()
            ])
            # 第二种颜色抖动变换（参数略有不同）
            self.transform_aug2 = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.2,      # 亮度抖动 ±20%
                    contrast=0.1,        # 对比度抖动 ±10%
                    saturation=0.2,      # 饱和度抖动 ±20%
                    hue=0.05             # 色调抖动 ±15%
                ),
                transforms.ToTensor()
            ])
        elif is_training:
            # 原来的单一数据增强
            self.transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.1,      # 亮度抖动 ±10%
                    contrast=0.1,        # 对比度抖动 ±10%
                    saturation=0.1,      # 饱和度抖动 ±10%
                    hue=0.1              # 色调抖动 ±5%
                ),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        # 获取所有图片文件名
        self.img_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.img_files.sort(key=lambda x: int(x.split('.')[0]))
        
    def __len__(self):
        # 如果启用3倍数据增强，返回3倍长度
        if self.use_3x_augmentation:  # 修改条件：只要启用3倍数据增强就返回3倍长度
            return len(self.img_files) * 3
        else:
            return len(self.img_files)
    
    def __getitem__(self, idx):
        # 如果启用3倍数据增强，需要计算实际的图片索引和增强类型
        if self.use_3x_augmentation:  # 修改条件：只要启用3倍数据增强就进行索引计算
            # 计算实际的图片索引
            actual_idx = idx // 3
            # 计算增强类型 (0: 原图, 1: 第一种增强, 2: 第二种增强)
            aug_type = idx % 3
        else:
            actual_idx = idx
            aug_type = 0
        
        img_name = self.img_files[actual_idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.json'))
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小
        image = cv2.resize(image, self.img_size)
        
        # 生成掩码
        mask = self.create_mask_from_json(mask_path, image.shape[0], image.shape[1])
        
        # 转换为PIL图像以应用颜色抖动
        image = Image.fromarray(image)
        
        # 根据增强类型应用不同的变换
        if self.use_3x_augmentation:  # 修改条件：只要启用3倍数据增强就应用对应变换
            if aug_type == 0:
                # 原图
                image = self.transform_original(image)
            elif aug_type == 1:
                # 第一种颜色抖动
                image = self.transform_aug1(image)
            else:  # aug_type == 2
                # 第二种颜色抖动
                image = self.transform_aug2(image)
        else:
            # 使用原来的变换方式
            image = self.transform(image)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def create_mask_from_json(self, json_path, height, width):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 创建空掩码，0表示正常皮肤，1表示受影响区域，2表示瘢痕主体
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 创建PIL图像用于绘制多边形
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        
        # 首先绘制KELOID_BOUNDARY区域
        for shape in data['shapes']:
            if shape['label'] == 'KELOID_BOUNDARY':
                points = [(int(p[0] * width / data['imageWidth']), 
                          int(p[1] * height / data['imageHeight'])) 
                          for p in shape['points']]
                draw.polygon(points, fill=1)
        
        # 将KELOID_BOUNDARY区域的值设为1
        boundary_mask = np.array(img)
        mask[boundary_mask == 1] = 1
        
        # 创建新的PIL图像用于绘制KELOID_BODY
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        
        # 绘制KELOID_BODY区域
        for shape in data['shapes']:
            if shape['label'] == 'KELOID_BODY':
                points = [(int(p[0] * width / data['imageWidth']), 
                          int(p[1] * height / data['imageHeight'])) 
                          for p in shape['points']]
                draw.polygon(points, fill=1)
        
        # 将KELOID_BODY区域的值设为2
        body_mask = np.array(img)
        mask[body_mask == 1] = 2
        
        return mask

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', model_name='best_unet_model.pth', patience=7):
    model.to(device)
    best_val_loss = float('inf')
    
    # 早停相关变量
    patience_counter = 0  # 耐心计数器
    early_stopped = False  # 是否触发早停
    
    # 创建训练历史记录字典
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'early_stopping': {
            'patience': patience,
            'triggered': False,
            'stopped_at_epoch': 0
        }
    }
    
    # 获取实际的训练集和验证集大小
    train_size = len(train_loader.sampler)
    val_size = len(val_loader.sampler)
    
    print(f"开始训练，共{num_epochs}个epochs，早停耐心值: {patience}")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - 训练阶段")
        train_pbar = tqdm(train_loader, desc=f"训练", leave=False)
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss = train_loss / train_size
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs} - 验证阶段")
        val_pbar = tqdm(val_loader, desc=f"验证", leave=False)
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss = val_loss / val_size
        
        # 记录训练历史
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(round(train_loss, 6))
        history['val_loss'].append(round(val_loss, 6))
        
        # 计算本轮耗时
        epoch_time = time.time() - epoch_start_time
        
        # 早停机制检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch + 1
            history['best_val_loss'] = round(val_loss, 6)
            torch.save(model.state_dict(), model_name)
            patience_counter = 0  # 重置耐心计数器
            print(f'Epoch {epoch+1}/{num_epochs} - 耗时: {epoch_time:.1f}秒, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f} ⭐ 新最佳!')
            print(f"保存新的最佳模型，验证损失: {val_loss:.4f}, 保存为: {model_name}")
        else:
            patience_counter += 1  # 增加耐心计数器
            print(f'Epoch {epoch+1}/{num_epochs} - 耗时: {epoch_time:.1f}秒, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f} (无改善 {patience_counter}/{patience})')
        
        # 检查是否触发早停
        if patience_counter >= patience:
            early_stopped = True
            history['early_stopping']['triggered'] = True
            history['early_stopping']['stopped_at_epoch'] = epoch + 1
            print(f"\n🛑 早停触发! 验证损失连续{patience}个epoch未改善，在第{epoch+1}个epoch停止训练")
            print(f"📊 最佳模型来自第{history['best_epoch']}个epoch，验证损失: {history['best_val_loss']:.6f}")
            break
    
    if not early_stopped:
        print(f"\n✅ 训练完成，共进行了{num_epochs}个epoch")
    
    return model, history

# 预测函数
def predict(model, image_path, device='cuda'):
    model.eval()
    
    # 加载并预处理图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])
    image = cv2.resize(image, (256, 256))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # 将预测结果调整回原始大小
    pred = cv2.resize(pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    
    return pred

def main():
    # 获取开始运行时的时间戳
    start_timestamp = datetime.now()
    timestamp_str = start_timestamp.strftime("%Y%m%d_%H%M%S")
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='U-Net瘢痕分割模型训练')
    parser.add_argument('-Trans', type=int, default=0, choices=[0, 1], 
                       help='是否启用数据增强: 0=不启用, 1=启用 (默认: 0)')
    args = parser.parse_args()
    
    # 根据参数决定是否使用数据增强和模型名称（加上时间戳）
    use_transform = bool(args.Trans)
    use_3x_augmentation = use_transform  # 当启用数据增强时，同时启用3倍数据增强
    
    if use_transform:
        model_name = f'best_trans_unet_model_{timestamp_str}.pth'
        final_model_name = f'final_trans_unet_model_{timestamp_str}.pth'
        history_file = f'training_history_trans_{timestamp_str}.json'
        print(f'数据增强模式: 启用3倍数据增强 (原图 + 两种颜色抖动)')
    else:
        model_name = f'best_unet_model_{timestamp_str}.pth'
        final_model_name = f'final_unet_model_{timestamp_str}.pth'
        history_file = f'training_history_{timestamp_str}.json'
        print(f'数据增强模式: 不启用数据增强')
    
    print(f'训练开始时间: {start_timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'最佳模型将保存为: {model_name}')
    print(f'训练历史将保存为: {history_file}')
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备并显示GPU信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    if device.type == 'cuda':
        print(f'当前使用的GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
        print(f'当前可用GPU内存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB')
    else:
        print('警告: 未检测到可用的GPU，将使用CPU进行训练，这可能会很慢')
    
    # 数据目录
    data_dir = 'dataset/final_labeled'
    print(f'加载数据集: {data_dir}')
    
    # 创建数据集时指定是否为训练集
    print('正在准备数据集...')
    full_dataset = KeloidDataset(data_dir, is_training=False, use_3x_augmentation=False)  # 用于获取原始数据集大小
    
    # 分割数据集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42
    )
    
    # 创建训练和验证数据集
    train_dataset = KeloidDataset(data_dir, is_training=use_transform, use_3x_augmentation=use_3x_augmentation)   # 训练集根据参数决定是否启用3倍数据增强
    val_dataset = KeloidDataset(data_dir, is_training=False, use_3x_augmentation=False)                            # 验证集固定不做数据增强
    
    # 计算实际的训练集和验证集大小
    if use_3x_augmentation:
        actual_train_size = len(train_indices) * 3
        actual_val_size = len(val_indices)
        print(f'原始数据集大小: {len(full_dataset)}个样本')
        print(f'3倍数据增强后 (仅训练集包含颜色抖动):')
        print(f'  训练集: {len(train_indices)} × 3 = {actual_train_size}个样本')
        print(f'  验证集: {len(val_indices)}个样本 (不增强)')
        print(f'  总计样本数: 训练{actual_train_size} + 验证{actual_val_size} = {actual_train_size + actual_val_size}')
    else:
        actual_train_size = len(train_indices)
        actual_val_size = len(val_indices)
        print(f'数据集大小: {len(full_dataset)}个样本')
        print(f'训练集: {len(train_indices)}个样本, 验证集: {len(val_indices)}个样本')
    
    # 需要调整数据加载器的采样器以适应3倍数据增强
    if use_3x_augmentation:
        # 仅训练集使用3倍数据增强索引
        train_indices_3x = []
        for idx in train_indices:
            train_indices_3x.extend([idx * 3, idx * 3 + 1, idx * 3 + 2])
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices_3x)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # 创建数据加载器
    print('创建数据加载器...')
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler)
    
    # 创建模型
    print('初始化U-Net模型...')
    model = UNet()
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数总数: {total_params:,}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print('开始训练过程...')
    train_start = time.time()
    model, training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=100, device=device, model_name=model_name, patience=20
    )
    
    total_training_time = time.time() - train_start
    print(f'训练完成，总耗时: {total_training_time//60}分{total_training_time%60:.1f}秒')
    
    # 添加训练信息到历史记录
    training_history['start_time'] = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    training_history['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_history['timestamp'] = timestamp_str
    training_history['total_training_time_seconds'] = round(total_training_time, 2)
    training_history['total_training_time_formatted'] = f'{total_training_time//60}分{total_training_time%60:.1f}秒'
    training_history['data_augmentation'] = use_transform
    training_history['use_3x_augmentation'] = use_3x_augmentation  # 新增记录
    training_history['original_dataset_size'] = len(full_dataset)
    training_history['dataset_size'] = actual_train_size + actual_val_size
    training_history['train_size'] = actual_train_size
    training_history['val_size'] = actual_val_size
    training_history['model_files'] = {
        'best_model': model_name,
        'final_model': final_model_name
    }
    
    # 保存训练历史到JSON文件
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    print(f'训练历史已保存为: {history_file}')
    
    # 保存最终模型
    torch.save(model.state_dict(), final_model_name)
    print(f'最终模型已保存为: {final_model_name}')
    
    # 打印训练总结
    print(f'\n训练总结:')
    print(f'开始时间: {training_history["start_time"]}')
    print(f'结束时间: {training_history["end_time"]}')
    print(f'最佳验证损失: {training_history["best_val_loss"]:.6f} (第{training_history["best_epoch"]}轮)')
    print(f'最终训练损失: {training_history["train_loss"][-1]:.6f}')
    print(f'最终验证损失: {training_history["val_loss"][-1]:.6f}')
    
    # 添加早停信息
    if training_history['early_stopping']['triggered']:
        print(f'🛑 早停触发: 在第{training_history["early_stopping"]["stopped_at_epoch"]}轮停止')
        print(f'📊 实际训练轮数: {len(training_history["epochs"])}/{100}')
        print(f'⏱️ 节省时间: 避免了{100-len(training_history["epochs"])}轮无效训练')
    else:
        print(f'✅ 完成全部训练: {len(training_history["epochs"])}/{100}轮')
        print(f'💡 提示: 模型可能还有改善空间，可以考虑增加epochs或调整patience')

if __name__ == '__main__':
    main()
