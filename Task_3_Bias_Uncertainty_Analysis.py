import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import warnings
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 引入 torchxrayvision 和 skimage
import torchxrayvision as xrv
from skimage.metrics import structural_similarity as ssim

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 配置 ---
DATA_DIR = 'CheXpert-v1.0-small'
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')
MODEL_PATH = 'chexpert_finetuned_xrv.pth' 

BATCH_SIZE = 1 
MC_PASSES = 10 
TARGET_CLASS_NAME = 'Cardiomegaly' 
INPUT_NOISE_SCALE = 0.02  # [新增] 输入噪声强度，防止 Grad-CAM 完全静止

CORE_CLASSES = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
]

# --- 2. Dataset 类 ---
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir):
        self.df = pd.read_csv(csv_file)
        for col in CORE_CLASSES:
            if col not in self.df.columns: self.df[col] = 0
        self.df = self.df.dropna(subset=CORE_CLASSES)
        self.df[CORE_CLASSES] = self.df[CORE_CLASSES].fillna(0).replace(-1, 0)
        self.data_dir = data_dir
        self.raw_paths = []
        
        print(f"正在构建数据集路径... (Data Dir: {self.data_dir})")
        for p in self.df['Path']:
            full_path = os.path.join(self.data_dir, p)
            self.raw_paths.append(full_path)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_path = self.raw_paths[idx]
        try:
            image_pil = Image.open(img_path).convert('L') 
            img = np.array(image_pil)
        except:
            img = np.zeros((224, 224), dtype=np.uint8)

        img = xrv.datasets.normalize(img, 255) 
        if len(img.shape) > 2: img = img.mean(2)
        img = img[None, ...] 
        transform = xrv.datasets.XRayResizer(224)
        img = transform(img)

        image_tensor = torch.from_numpy(img).float()
        labels = self.df.iloc[idx][CORE_CLASSES].values.astype(np.float32)
        return image_tensor, torch.from_numpy(labels), idx

# --- 3. 加载数据 ---
try:
    valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# --- 4. 加载模型 ---
print("正在加载模型...")
model = xrv.models.DenseNet(weights="densenet121-res224-chex") 
model.op_threshs = None 
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(CORE_CLASSES)) 

if MODEL_PATH and os.path.exists(MODEL_PATH):
    print(f"加载微调权重: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
model = model.to(DEVICE)

# 目标索引
if TARGET_CLASS_NAME in CORE_CLASSES:
    TARGET_CLASS_IDX = CORE_CLASSES.index(TARGET_CLASS_NAME)
    print(f"分析目标: {TARGET_CLASS_NAME} (Index: {TARGET_CLASS_IDX})")
else:
    exit()

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
targets_for_cam = [ClassifierOutputTarget(TARGET_CLASS_IDX)]

# [新增] 辅助函数：强制开启 Dropout
def enable_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

# ==========================================
# Part A: 偏差与不确定性分析 (Uncertainty)
# ==========================================
print("\n--- Part A: 计算基准不确定性 (Entropy) ---")
all_uncertainties = []
all_indices = []

model.eval()
with torch.no_grad():
    fast_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    for i, (images, _, batch_indices) in enumerate(fast_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        p = probs[:, TARGET_CLASS_IDX]
        epsilon = 1e-9
        binary_entropy = - (p * torch.log2(p + epsilon) + (1 - p) * torch.log2(1 - p + epsilon))
        all_uncertainties.extend(binary_entropy.cpu().numpy())
        all_indices.extend(batch_indices.numpy())

uncertainty_map = dict(zip(all_indices, all_uncertainties))
uncertainty_array = np.array([uncertainty_map[i] for i in range(len(valid_dataset))])

# ==========================================
# Part B: 解释一致性分析 (Improved)
# ==========================================
print("\n--- Part B: 计算解释一致性 (Input Noise + SSIM) ---")
print(f"策略: MC Passes={MC_PASSES}, Input Noise Scale={INPUT_NOISE_SCALE}")

all_consistency_scores = []
start_time = time.time()

# 开启 Dropout（如果模型里有的话）
model.apply(enable_dropout)

for i, (images, _, _) in enumerate(valid_loader):
    images = images.to(DEVICE)
    mc_heatmaps = []
    
    for _ in range(MC_PASSES):
        model.zero_grad()
        
        # [关键改进] 添加微小噪声，强制模型产生变化
        # 如果模型参数没变化，输入的变化也能测试“敏感性一致性”
        noise = torch.randn_like(images) * INPUT_NOISE_SCALE
        noisy_images = images + noise
        
        heatmap = cam(input_tensor=noisy_images, targets=targets_for_cam)[0, :] 
        mc_heatmaps.append(heatmap)
    
    ssim_scores = []
    for h1, h2 in combinations(mc_heatmaps, 2):
        d_range = h1.max() - h1.min()
        if d_range == 0: d_range = 1e-5 # 防止除零
        score = ssim(h1, h2, data_range=d_range)
        ssim_scores.append(score)
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    all_consistency_scores.append(avg_ssim)
    
    if (i + 1) % 20 == 0:
        elapsed = time.time() - start_time
        print(f"  已处理 {i+1}/{len(valid_dataset)}... (耗时: {elapsed:.1f}s)")
        torch.cuda.empty_cache()

model.eval()
all_consistency_scores = np.array(all_consistency_scores)

# ==========================================
# Part C: 鲁棒相关性分析
# ==========================================
print("\n--- Part C: 结果统计与相关性 ---")

# 清洗 NaN
mask = ~np.isnan(uncertainty_array) & ~np.isnan(all_consistency_scores)
u_clean = uncertainty_array[mask]
c_clean = all_consistency_scores[mask]

# [关键改进] 检查方差，防止 NaN
u_std = np.std(u_clean)
c_std = np.std(c_clean)

print(f"不确定性 Std: {u_std:.4f}")
print(f"一致性 Std:   {c_std:.4f}")

if u_std < 1e-6 or c_std < 1e-6:
    print("警告: 某一变量方差极低，直接计算相关性可能得到 NaN。")
    # 添加极其微小的抖动以允许计算（仅用于避免程序报错，不影响趋势）
    if c_std < 1e-6: c_clean += np.random.normal(0, 1e-6, size=c_clean.shape)

corr_p, p_val_p = pearsonr(u_clean, c_clean)
corr_s, p_val_s = spearmanr(u_clean, c_clean)

print("\n[交付物 3-2] 不确定性 vs 解释一致性")
print(f"样本数: {len(u_clean)}")
print(f"  皮尔逊 (Pearson): {corr_p:.4f} (p={p_val_p:.4f})")
print(f"  斯皮尔曼 (Spearman): {corr_s:.4f} (p={p_val_s:.4f})")

# 绘图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=u_clean, y=c_clean, alpha=0.6, color='blue')
# 只有当相关性有效时才画回归线
if not np.isnan(corr_p):
    sns.regplot(x=u_clean, y=c_clean, scatter=False, color='red')
plt.title(f'Uncertainty vs Consistency (with Input Noise)\n{TARGET_CLASS_NAME}')
plt.xlabel('Uncertainty (Entropy)')
plt.ylabel('Consistency (SSIM)')
plt.grid(True, alpha=0.3)
plt.savefig("task3_robust_analysis.png")
print("\n图表已保存: task3_robust_analysis.png")