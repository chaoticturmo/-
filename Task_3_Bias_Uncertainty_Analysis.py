import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import os
import numpy as np
import warnings
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import time


from skimage.metrics import structural_similarity as ssim

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = 'CheXpert-v1.0-small'
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')
MODEL_PATH = 'chexpert_model.pth'

BATCH_SIZE = 1 
CLASSES = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
]
NUM_CLASSES = len(CLASSES)
TARGET_CLASS_IDX = 0 
targets_for_cam = [ClassifierOutputTarget(TARGET_CLASS_IDX)]


MC_PASSES = 10


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file).dropna(subset=CLASSES)
        self.data_dir = data_dir
        self.transform = transform
        self.df[CLASSES] = self.df[CLASSES].fillna(0).replace(-1, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        labels = self.df.iloc[idx][CLASSES].values.astype(np.float32)
        return image_tensor, torch.from_numpy(labels), idx

def load_model(path, num_classes):
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),  
        nn.Linear(num_features, num_classes) 
    )
   

    map_loc = torch.device('cpu') if not torch.cuda.is_available() else None
    
    
    try:
        state_dict = torch.load(path, map_location=map_loc)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if k.startswith('classifier.'):
                new_k = k.replace('classifier.', 'classifier.1.')
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        
    except Exception as e:
        print(f"错误: 加载权重模型失败: {e}")
        
    return model


data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR, transform=data_transform)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = load_model(MODEL_PATH, NUM_CLASSES).to(DEVICE)
cam = GradCAM(model=model, target_layers=[model.features.norm5])

print("--- Part A: 偏差与不确定性分析 ---")
print("计算所有样本的不确定性 (熵)...")

all_uncertainties = []
all_indices = []

model.eval() 
with torch.no_grad():
    fast_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    for (images, _, batch_indices) in fast_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        p = torch.sigmoid(outputs)
        epsilon = 1e-9
        binary_entropy = - (p * torch.log2(p + epsilon) + (1 - p) * torch.log2(1 - p + epsilon))
        sample_uncertainty = torch.mean(binary_entropy, dim=1) # (N,)
        
        all_uncertainties.extend(sample_uncertainty.cpu().numpy())
        all_indices.extend(batch_indices.numpy())

uncertainty_map = dict(zip(all_indices, all_uncertainties))
uncertainty_array = np.array([uncertainty_map[i] for i in range(len(valid_dataset))])


metadata_df = valid_dataset.df.reset_index(drop=True)
metadata_df['Age'] = pd.to_numeric(metadata_df['Age'], errors='coerce')
metadata_df = metadata_df.dropna(subset=['Age'])
bins = [0, 40, 65, np.inf]
labels = ['<=40', '41-65', '>65']
metadata_df['age_group'] = pd.cut(metadata_df['Age'], bins=bins, labels=labels, right=True)

metadata_df['uncertainty'] = uncertainty_array[metadata_df.index]

print("\n--- [交付物]: 各群体平均不确定性 ---")
print(f"  Overall Mean Uncertainty: {metadata_df['uncertainty'].mean():.4f}\n")

print("  --- 按性别 (Sex) ---")
print(metadata_df.groupby('Sex')['uncertainty'].agg(['mean', 'std', 'count']))

print("\n  --- 按年龄 (age_group) ---")
print(metadata_df.groupby('age_group')['uncertainty'].agg(['mean', 'std', 'count']))

print("\n--- Part B: 不确定性与解释一致性相关性 ---")

all_consistency_scores = []
start_time = time.time()

model.train() 

for i, (images, _, _) in enumerate(valid_loader):
    images = images.to(DEVICE) 
    
    mc_heatmaps = []
    
    torch.set_grad_enabled(True) 
    for _ in range(MC_PASSES):
        heatmap = cam(input_tensor=images, targets=targets_for_cam)[0, :] 
        mc_heatmaps.append(heatmap)
    torch.set_grad_enabled(False)
    
    ssim_scores = []
    for h1, h2 in combinations(mc_heatmaps, 2):
        score = ssim(h1, h2, data_range=h1.max() - h1.min())
        ssim_scores.append(score)
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 1.0 
    all_consistency_scores.append(avg_ssim)
    
    if (i + 1) % 20 == 0:
        elapsed = time.time() - start_time
        print(f"  已处理 {i+1}/{len(valid_dataset)}... (耗时: {elapsed:.1f}s)")

model.eval() 
all_consistency_scores = np.array(all_consistency_scores)


print("\n--- DEBUGGING NANs ---")
print("--- 检查 'uncertainty_array' ---")
print(f"  Mean: {np.mean(uncertainty_array)}")
print(f"  Std Dev (方差): {np.std(uncertainty_array)}")
print(f"  Has NaNs: {np.isnan(uncertainty_array).any()}")
print(f"  First 5 values: {uncertainty_array[:5]}")

print("\n--- 检查 'all_consistency_scores' ---")
print(f"  Mean: {np.mean(all_consistency_scores)}")
print(f"  Std Dev (方差): {np.std(all_consistency_scores)}")
print(f"  NaN count: {np.sum(np.isnan(all_consistency_scores))}")
print(f"  First 5 values: {all_consistency_scores[:5]}")
print("--------------------------\n")

print("计算完成。")

corr_p, p_val_p = pearsonr(uncertainty_array, all_consistency_scores)
corr_s, p_val_s = spearmanr(uncertainty_array, all_consistency_scores)

print("\n--- [交付物]: 不确定性 vs 解释一致性 ---")
print(f"不确定性 (熵) vs. 解释一致性 (MC-SSIM)")
print(f"  皮尔逊相关系数 (Pearson): {corr_p:.4f} (p-value: {p_val_p:.4f})")
print(f"  斯皮尔曼相关系数 (Spearman): {corr_s:.4f} (p-value: {p_val_s:.4f})")
print("----------------------------------------------------------")

print("\n任务3 分析完成。")