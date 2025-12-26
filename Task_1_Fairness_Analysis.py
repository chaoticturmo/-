import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import warnings
from PIL import Image

# --- 引入 torchxrayvision ---
import torchxrayvision as xrv
# 需要 scikit-image 来进行 XRV 推荐的 resize
import skimage 

warnings.filterwarnings('ignore')

# --- 配置 ---
# 请确保这里指向包含 valid.csv 的文件夹
DATA_DIR = 'CheXpert-v1.0-small' 
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')
BATCH_SIZE = 32

# 你的目标 5 个类别
TARGET_CLASSES = [
    'Cardiomegaly', 
    'Edema', 
    'Consolidation', 
    'Atelectasis', 
    'Pleural Effusion'
]
NUM_CLASSES = len(TARGET_CLASSES)

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir):
        self.df = pd.read_csv(csv_file)
        
        # 1. 确保目标列存在
        for col in TARGET_CLASSES:
            if col not in self.df.columns:
                self.df[col] = 0
        
        self.df = self.df.dropna(subset=TARGET_CLASSES)
        
        # 2. 处理标签: NaN -> 0, -1 (Uncertain) -> 0 (Negative)
        self.df[TARGET_CLASSES] = self.df[TARGET_CLASSES].fillna(0)
        self.df[TARGET_CLASSES] = self.df[TARGET_CLASSES].replace(-1, 0)
        
        self.data_dir = data_dir
        self.raw_paths = []
        
        # 3. 智能路径处理 (防止 CheXpert/CheXpert 双层目录问题)
        print(f"正在构建数据集路径... (Data Dir: {self.data_dir})")
        # 检查 CSV 中的第一条路径是否已经包含 Data Dir
        if not self.df.empty:
            first_path = self.df.iloc[0]['Path'] # e.g., CheXpert-v1.0-small/valid/...
            full_test_path = os.path.join(self.data_dir, first_path)
            if not os.path.exists(full_test_path) and first_path.startswith(self.data_dir):
                # 如果直接拼接找不到，且 csv 路径里本身就包含 data_dir 的名字
                # 说明 data_dir 可能就是根目录
                print("检测到路径可能重复，尝试调整...")
                self.use_absolute_join = False
            else:
                self.use_absolute_join = True
                
        for p in self.df['Path']:
            if self.use_absolute_join:
                full_path = os.path.join(self.data_dir, p)
            else:
                # 如果 CSV 路径已经是完整的相对路径，且 data_dir 是当前目录
                full_path = p 
            self.raw_paths.append(full_path)

        if len(self.raw_paths) > 0:
            print(f"路径示例: {self.raw_paths[0]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.raw_paths[idx]
        
        # --- [关键修改] XRV 专用预处理 ---
        try:
            # 1. 读取为灰度图 (L)
            image_pil = Image.open(img_path).convert('L') 
            img = np.array(image_pil)
        except Exception as e:
            # 异常处理：返回全黑图防止崩溃
            print(f"Error reading {img_path}: {e}")
            img = np.zeros((224, 224), dtype=np.uint8)

        # 2. XRV 官方归一化: 将 0-255 映射到 [-1024, 1024]
        img = xrv.datasets.normalize(img, 255) 

        # 3. 调整维度 (H, W) -> (1, H, W)
        if len(img.shape) > 2: 
            img = img.mean(2) # 如果意外读到了 RGB
        img = img[None, ...] # 增加 Channel 维度

        # 4. XRV 官方 Resize (保持比例 + 填充黑边)
        # 这比 CenterCrop 更好，不会切掉肺部边缘
        transform = xrv.datasets.XRayResizer(224)
        img = transform(img)

        # 5. 转 Tensor
        image_tensor = torch.from_numpy(img).float()
        
        labels = self.df.iloc[idx][TARGET_CLASSES].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        
        return image_tensor, labels

# --- 不再需要单独定义 data_transform，逻辑已移入 Dataset ---

# 检查数据文件
try:
    # 实例化 Dataset
    valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
except Exception as e:
    print(f"数据加载初始化失败: {e}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 加载 XRV 模型 ---
print("正在加载 torchxrayvision 预训练模型 (densenet121-res224-chex)...")
model = xrv.models.DenseNet(weights="densenet121-res224-chex")
model.to(device)
model.eval()

print(f"XRV 模型加载完毕。模型支持的病理: {model.pathologies}")

# --- 类别对齐逻辑 ---
# 定义映射字典：{ 你的CSV列名 : XRV模型输出名 }
LABEL_MAPPING = {
    'Pleural Effusion': 'Effusion'  # XRV 使用 'Effusion'
}

target_indices = []
print("\n--- 类别索引映射 ---")
for target in TARGET_CLASSES:
    # 1. 尝试直接匹配
    if target in model.pathologies:
        idx = model.pathologies.index(target)
        target_indices.append(idx)
        print(f"Mapping '{target}' -> Index {idx}")
    
    # 2. 尝试通过字典映射匹配
    elif target in LABEL_MAPPING and LABEL_MAPPING[target] in model.pathologies:
        mapped_name = LABEL_MAPPING[target]
        idx = model.pathologies.index(mapped_name)
        target_indices.append(idx)
        print(f"Mapping '{target}' (via '{mapped_name}') -> Index {idx}")
        
    # 3. 匹配失败
    else:
        print(f"警告: 模型输出中找不到 '{target}'")
        target_indices.append(-1)

print(f"\n在 {device} 上运行评估...")

all_preds = []
all_labels = []

# 开始推理
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        
        # XRV forward 输出 Logits
        outputs = model(images)
        
        # 如果模型有 op_threshs (操作点阈值)，XRV 有时会自动处理，
        # 但 densenet121-res224-chex 通常输出 raw logits。
        # 我们手动 Sigmoid 确保归一化到 0-1
        preds = torch.sigmoid(outputs)
        
        # 筛选出我们关心的 5 列
        batch_preds = []
        for idx in target_indices:
            if idx != -1:
                batch_preds.append(preds[:, idx].view(-1, 1))
            else:
                batch_preds.append(torch.zeros(images.size(0), 1).to(device))
        
        batch_preds = torch.cat(batch_preds, dim=1)
        
        all_preds.append(batch_preds.cpu().numpy())
        all_labels.append(labels.numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"已处理 {batch_idx + 1} / {len(valid_loader)} batches...")

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"已完成对 {len(all_labels)} 个样本的预测。")

# --- 分层分析 (保持不变) ---
metadata_df = valid_dataset.df.reset_index(drop=True)
# 确保长度对齐（如果 dataset 内部有 dropna，这里需要重新对齐）
if len(metadata_df) != len(all_preds):
    print("警告: 元数据长度与预测结果长度不一致，正在截断对齐...")
    metadata_df = metadata_df.iloc[:len(all_preds)]

metadata_df['Age'] = pd.to_numeric(metadata_df['Age'], errors='coerce')
metadata_df = metadata_df.dropna(subset=['Age']) 

bins = [0, 40, 65, np.inf]
labels = ['<=40', '41-65', '>65']
metadata_df['age_group'] = pd.cut(metadata_df['Age'], bins=bins, labels=labels, right=True)

# 重新获取索引（因为上面可能有 dropna）
valid_indices = metadata_df.index

subgroups = {
    'all': valid_indices, 
    'gender_male': metadata_df[metadata_df['Sex'] == 'Male'].index,
    'gender_female': metadata_df[metadata_df['Sex'] == 'Female'].index,
    'age_<=40': metadata_df[metadata_df['age_group'] == '<=40'].index,
    'age_41-65': metadata_df[metadata_df['age_group'] == '41-65'].index,
    'age_>65': metadata_df[metadata_df['age_group'] == '>65'].index,
}

results = {}

print("\nTask-1")
print("--- 1. 分层性能计算 [cite: 18] ---")

for class_idx, class_name in enumerate(TARGET_CLASSES):
    print(f"--- 正在分析病症: {class_name} ---")
    
    class_results = {}
    
    for group_name, indices in subgroups.items():
        if len(indices) == 0:
            continue
            
        # 注意：这里需要用 indices 映射回 all_labels 的正确位置
        # 由于我们之前可能对 metadata_df 做了 dropna，这里需要小心索引对齐
        # 最稳妥的方法是使用 metadata_df 的整数位置索引 (iloc)
        # 但为了简单，假设上面的 dropna 很少
        
        # 获取当前子群体在 all_preds 中的对应行
        # all_preds 和 metadata_df 现在是行对齐的
        current_labels = all_labels[indices, class_idx]
        current_preds = all_preds[indices, class_idx]
        
        if len(np.unique(current_labels)) < 2:
            # print(f"  Skipping {group_name}: 只有一种标签")
            continue

        try:
            auc = roc_auc_score(current_labels, current_preds)
            # F1 需要阈值，这里暂时用 0.5
            f1 = f1_score(current_labels, (current_preds > 0.5).astype(int))
            
            class_results[group_name] = {'auc': auc, 'f1': f1}
            print(f"  Group: {group_name:<15} | N={len(indices):<5} | AUC: {auc:.4f} | F1: {f1:.4f}")
        except ValueError:
            pass
    
    results[class_name] = class_results

print("\n--- 2. 性能差异计算 [cite: 21] ---")
for class_name, class_results in results.items():
    print(f"--- {class_name} 的差异 (Delta) ---")
    
    if 'gender_male' in class_results and 'gender_female' in class_results:
        delta_auc = abs(class_results['gender_male']['auc'] - class_results['gender_female']['auc'])
        delta_f1 = abs(class_results['gender_male']['f1'] - class_results['gender_female']['f1'])
        print(f"  性别差异 (Male vs Female): ΔAUC = {delta_auc:.4f}, ΔF1 = {delta_f1:.4f}")

    if 'age_<=40' in class_results and 'age_>65' in class_results:
        delta_auc = abs(class_results['age_<=40']['auc'] - class_results['age_>65']['auc'])
        delta_f1 = abs(class_results['age_<=40']['f1'] - class_results['age_>65']['f1'])
        print(f"  年龄差异 (<=40 vs >65): ΔAUC = {delta_auc:.4f}, ΔF1 = {delta_f1:.4f}")

print("\n任务1 分析完成。")