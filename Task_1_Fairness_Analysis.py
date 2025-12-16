import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


DATA_DIR = 'CheXpert-v1.0-small'
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')
MODEL_PATH = 'chexpert_model.pth'

BATCH_SIZE = 32
CLASSES = [
    'Cardiomegaly', 
    'Edema', 
    'Consolidation', 
    'Atelectasis', 
    'Pleural Effusion'
]
NUM_CLASSES = len(CLASSES)


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.dropna(subset=CLASSES)
        self.data_dir = data_dir
        self.transform = transform
        self.df[CLASSES] = self.df[CLASSES].fillna(0)
        self.df[CLASSES] = self.df[CLASSES].replace(-1, 0)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = self.df.iloc[idx][CLASSES].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        return image, labels

def load_model(path, num_classes):
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    
  
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(path))
        
    return model

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR, transform=data_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
except FileNotFoundError:
    print(f"错误: 无法在 '{DATA_DIR}' 找到 'valid.csv'")
   
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, NUM_CLASSES)
model.to(device)
model.eval() 

print(f"模型 {MODEL_PATH} 加载完毕。在 {device} 上运行评估...")


all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())


all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"已完成对 {len(all_labels)} 个样本的预测。")


metadata_df = valid_dataset.df.reset_index(drop=True)

assert len(metadata_df) == len(all_preds), "元数据和预测结果长度不匹配!"


metadata_df['Age'] = pd.to_numeric(metadata_df['Age'], errors='coerce')
metadata_df = metadata_df.dropna(subset=['Age']) 

bins = [0, 40, 65, np.inf]
labels = ['<=40', '41-65', '>65']
metadata_df['age_group'] = pd.cut(metadata_df['Age'], bins=bins, labels=labels, right=True)

subgroups = {
    'all': metadata_df.index, 
    'gender_male': metadata_df[metadata_df['Sex'] == 'Male'].index,
    'gender_female': metadata_df[metadata_df['Sex'] == 'Female'].index,
    'age_<=40': metadata_df[metadata_df['age_group'] == '<=40'].index,
    'age_41-65': metadata_df[metadata_df['age_group'] == '41-65'].index,
    'age_>65': metadata_df[metadata_df['age_group'] == '>65'].index,
}

results = {}

print("\n--- 1. 分层性能计算 [cite: 18] ---")

for class_idx, class_name in enumerate(CLASSES):
    print(f"\n--- 正在分析病症: {class_name} ---")
    
    class_results = {}
    
    for group_name, indices in subgroups.items():
        if len(indices) == 0:
            print(f"  Skipping {group_name}: 0 samples")
            continue
            
        group_labels = all_labels[indices, class_idx]
        group_preds = all_preds[indices, class_idx]
        
        if len(np.unique(group_labels)) < 2:
            print(f"  Skipping {group_name}: 只有一种标签，无法计算 AUC/F1")
            continue

        auc = roc_auc_score(group_labels, group_preds)
 
        f1 = f1_score(group_labels, (group_preds > 0.5).astype(int))
        
        class_results[group_name] = {'auc': auc, 'f1': f1, 'count': len(indices)}
        
        print(f"  Group: {group_name:<15} | N={len(indices):<5} | AUC: {auc:.4f} | F1: {f1:.4f}")
    
    results[class_name] = class_results

print("\n--- 2. 性能差异计算 [cite: 21] ---")
for class_name, class_results in results.items():
    print(f"\n--- {class_name} 的差异 (Delta) ---")
    
    if 'gender_male' in class_results and 'gender_female' in class_results:
        delta_auc_gender = abs(class_results['gender_male']['auc'] - class_results['gender_female']['auc'])
        delta_f1_gender = abs(class_results['gender_male']['f1'] - class_results['gender_female']['f1'])
        print(f"  性别差异 (Male vs Female): ΔAUC = {delta_auc_gender:.4f}, ΔF1 = {delta_f1_gender:.4f}")

    if 'age_<=40' in class_results and 'age_>65' in class_results:
        delta_auc_age = abs(class_results['age_<=40']['auc'] - class_results['age_>65']['auc'])
        delta_f1_age = abs(class_results['age_<=40']['f1'] - class_results['age_>65']['f1'])
        print(f"  年龄差异 (<=40 vs >65): ΔAUC = {delta_auc_age:.4f}, ΔF1 = {delta_f1_age:.4f}")

print("\n任务1 分析完成。")