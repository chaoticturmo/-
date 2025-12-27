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
import torchxrayvision as xrv
import skimage 

warnings.filterwarnings('ignore')

DATA_DIR = 'CheXpert-v1.0-small' 
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')
BATCH_SIZE = 32

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
        
        for col in TARGET_CLASSES:
            if col not in self.df.columns:
                self.df[col] = 0
        
        self.df = self.df.dropna(subset=TARGET_CLASSES)
        
        self.df[TARGET_CLASSES] = self.df[TARGET_CLASSES].fillna(0)
        self.df[TARGET_CLASSES] = self.df[TARGET_CLASSES].replace(-1, 0)
        
        self.data_dir = data_dir
        self.raw_paths = []
        
        print(f"正在构建数据集路径... (Data Dir: {self.data_dir})")
        if not self.df.empty:
            first_path = self.df.iloc[0]['Path']
            full_test_path = os.path.join(self.data_dir, first_path)
            if not os.path.exists(full_test_path) and first_path.startswith(self.data_dir):
                print("检测到路径可能重复，尝试调整...")
                self.use_absolute_join = False
            else:
                self.use_absolute_join = True
                
        for p in self.df['Path']:
            if self.use_absolute_join:
                full_path = os.path.join(self.data_dir, p)
            else:
                full_path = p 
            self.raw_paths.append(full_path)

        if len(self.raw_paths) > 0:
            print(f"路径示例: {self.raw_paths[0]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.raw_paths[idx]
        
        try:
            image_pil = Image.open(img_path).convert('L') 
            img = np.array(image_pil)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            img = np.zeros((224, 224), dtype=np.uint8)

        img = xrv.datasets.normalize(img, 255) 

        if len(img.shape) > 2: 
            img = img.mean(2)
        img = img[None, ...]

        transform = xrv.datasets.XRayResizer(224)
        img = transform(img)

        image_tensor = torch.from_numpy(img).float()
        
        labels = self.df.iloc[idx][TARGET_CLASSES].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        
        return image_tensor, labels

try:
    valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
except Exception as e:
    print(f"数据加载初始化失败: {e}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("正在加载 torchxrayvision 预训练模型 (densenet121-res224-chex)...")
model = xrv.models.DenseNet(weights="densenet121-res224-chex")
model.to(device)
model.eval()

print(f"XRV 模型加载完毕。模型支持的病理: {model.pathologies}")

LABEL_MAPPING = {
    'Pleural Effusion': 'Effusion'
}

target_indices = []
print("\n--- 类别索引映射 ---")
for target in TARGET_CLASSES:
    if target in model.pathologies:
        idx = model.pathologies.index(target)
        target_indices.append(idx)
        print(f"Mapping '{target}' -> Index {idx}")
    elif target in LABEL_MAPPING and LABEL_MAPPING[target] in model.pathologies:
        mapped_name = LABEL_MAPPING[target]
        idx = model.pathologies.index(mapped_name)
        target_indices.append(idx)
        print(f"Mapping '{target}' (via '{mapped_name}') -> Index {idx}")
    else:
        print(f"警告: 模型输出中找不到 '{target}'")
        target_indices.append(-1)

print(f"\n在 {device} 上运行评估...")

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        
        outputs = model(images)
        
        preds = torch.sigmoid(outputs)
        
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

metadata_df = valid_dataset.df.reset_index(drop=True)
if len(metadata_df) != len(all_preds):
    print("警告: 元数据长度与预测结果长度不一致，正在截断对齐...")
    metadata_df = metadata_df.iloc[:len(all_preds)]

metadata_df['Age'] = pd.to_numeric(metadata_df['Age'], errors='coerce')
metadata_df = metadata_df.dropna(subset=['Age']) 

bins = [0, 40, 65, np.inf]
labels = ['<=40', '41-65', '>65']
metadata_df['age_group'] = pd.cut(metadata_df['Age'], bins=bins, labels=labels, right=True)

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
            
        current_labels = all_labels[indices, class_idx]
        current_preds = all_preds[indices, class_idx]
        
        if len(np.unique(current_labels)) < 2:
            continue

        try:
            auc = roc_auc_score(current_labels, current_preds)
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
