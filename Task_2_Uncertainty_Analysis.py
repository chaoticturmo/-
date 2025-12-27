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
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.transform 

import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = 'CheXpert-v1.0-small' 
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv') 

BATCH_SIZE = 16 
TARGET_CLASS_NAME = 'Atelectasis' 

CORE_CLASSES = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
]

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir):
        self.df = pd.read_csv(csv_file)
        
        for col in CORE_CLASSES:
            if col not in self.df.columns:
                self.df[col] = 0
        self.df = self.df.dropna(subset=CORE_CLASSES)
        self.df[CORE_CLASSES] = self.df[CORE_CLASSES].fillna(0).replace(-1, 0)
        
        self.data_dir = data_dir
        self.raw_paths = []
        
        print(f"正在构建数据集路径... (Data Dir: {self.data_dir})")
        for p in self.df['Path']:
            full_path = os.path.join(self.data_dir, p)
            self.raw_paths.append(full_path)

        if len(self.raw_paths) > 0:
            print(f"路径示例 (应为双层): {self.raw_paths[0]}")
            if not os.path.exists(self.raw_paths[0]):
                print(f"!!! 严重警告: 找不到文件 {self.raw_paths[0]}")
                print("请检查 DATA_DIR 是否需要改为 'CheXpert-v1.0-small/CheXpert-v1.0-small' ?")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.raw_paths[idx]
        
        try:
            image_pil = Image.open(img_path).convert('L') 
            img = np.array(image_pil)
        except Exception as e:
            print(f"[Error] 无法读取图片: {img_path}")
            img = np.zeros((224, 224), dtype=np.uint8)

        img = xrv.datasets.normalize(img, 255) 
        if len(img.shape) > 2: img = img.mean(2) 
        img = img[None, ...] 
        transform = xrv.datasets.XRayResizer(224)
        img = transform(img)

        image_tensor = torch.from_numpy(img).float()
        
        vis_img = (img.squeeze() + 1024) / 2048
        vis_img = np.clip(vis_img, 0, 1)
        vis_img = np.stack([vis_img]*3, axis=2) 
        vis_transform = transforms.ToTensor()
        vis_image_tensor = vis_transform(vis_img) 
        
        labels = self.df.iloc[idx][CORE_CLASSES].values.astype(np.float32)
        return image_tensor, torch.from_numpy(labels), vis_image_tensor, idx

try:
    valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
except Exception as e:
    print(f"数据加载初始化失败: {e}")
    exit()

print("正在加载 torchxrayvision 模型...")
model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(DEVICE)
for param in model.parameters():
    param.requires_grad = True
model.eval()

print("\n[Sanity Check] 检查模型输入数据分布...")
first_batch, _, _, _ = next(iter(valid_loader))
print(f"Min Value: {first_batch.min().item():.2f} (应接近 -1024)")
print(f"Max Value: {first_batch.max().item():.2f} (应接近 1024)")
if first_batch.min().item() == -1024 and first_batch.max().item() == -1024:
    print("!!! 错误: 图片依然是全黑的！请检查路径！")
    exit()
else:
    print("数据范围正常，继续分析...")
print("-" * 30)

target_layers = [model.features[-1]] 
LABEL_MAPPING = {
    'Pleural Effusion': 'Effusion',
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Edema': 'Edema',
    'Consolidation': 'Consolidation'
}

search_name = TARGET_CLASS_NAME
if search_name in model.pathologies:
    TARGET_CLASS_IDX = model.pathologies.index(search_name)
elif search_name in LABEL_MAPPING and LABEL_MAPPING[search_name] in model.pathologies:
    search_name = LABEL_MAPPING[search_name]
    TARGET_CLASS_IDX = model.pathologies.index(search_name)
else:
    print(f"错误: 模型不支持类别 '{TARGET_CLASS_NAME}'")
    exit()

print(f"分析目标: '{TARGET_CLASS_NAME}' (Mapped to '{search_name}', Index: {TARGET_CLASS_IDX})")
cam = GradCAM(model=model, target_layers=target_layers)

print(f"开始分析... (Batch Size: {BATCH_SIZE})")

all_uncertainties = []
all_dispersions = []
sample_indices = [] 
targets_for_cam = [ClassifierOutputTarget(TARGET_CLASS_IDX)]

count = 0
for (images, _, vis_images, batch_indices) in valid_loader:
    images = images.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(images) 
        probs = torch.sigmoid(outputs)
        p = probs[:, TARGET_CLASS_IDX]
        epsilon = 1e-9
        binary_entropy = - (p * torch.log2(p + epsilon) + (1 - p) * torch.log2(1 - p + epsilon))
        all_uncertainties.extend(binary_entropy.cpu().numpy())

    grayscale_cam = cam(input_tensor=images, targets=targets_for_cam) 
    dispersions = np.var(grayscale_cam, axis=(1, 2))
    all_dispersions.extend(dispersions)
    
    sample_indices.extend(batch_indices.numpy())
    
    count += len(images)
    if count % 100 == 0:
        print(f"  已处理 {count} / {len(valid_dataset)} 样本...")
        torch.cuda.empty_cache()

all_uncertainties = np.array(all_uncertainties)
all_dispersions = np.array(all_dispersions)

mask = ~np.isnan(all_uncertainties) & ~np.isnan(all_dispersions)
uncertainties_clean = all_uncertainties[mask]
dispersions_clean = all_dispersions[mask]

mean_entropy = np.mean(uncertainties_clean)
std_entropy = np.std(uncertainties_clean)
mean_disp = np.mean(dispersions_clean)
std_disp = np.std(dispersions_clean)

corr_p, p_val_p = pearsonr(uncertainties_clean, dispersions_clean)
corr_s, p_val_s = spearmanr(uncertainties_clean, dispersions_clean)

print("\n========== 任务2 分析结果 (修复后) ==========")
print(f"目标病症: {TARGET_CLASS_NAME}")
print(f"有效样本数: {len(uncertainties_clean)}")
print("-" * 30)
print(f"1. 不确定性 (Entropy): Mean={mean_entropy:.4f}, Std={std_entropy:.4f}")
print(f"2. 解释分散性 (Variance): Mean={mean_disp:.4f}, Std={std_disp:.4f}")
print("-" * 30)
print(f"3. 相关性分析:")
print(f"   皮尔逊 (Pearson): {corr_p:.4f} (p={p_val_p:.4f})")
print(f"   斯皮尔曼 (Spearman): {corr_s:.4f} (p={p_val_s:.4f})")
print("====================================")

plt.figure(figsize=(10, 6))
sns.histplot(uncertainties_clean, bins=30, kde=True, color='skyblue', edgecolor='black')
plt.axvline(mean_entropy, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_entropy:.3f}')
plt.title(f'Distribution of Uncertainty (Entropy) - {TARGET_CLASS_NAME}')
plt.savefig('task2_entropy_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(dispersions_clean, bins=30, kde=True, color='salmon', edgecolor='black')
plt.axvline(mean_disp, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_disp:.4f}')
plt.title(f'Distribution of Heatmap Variance - {TARGET_CLASS_NAME}')
plt.savefig('task2_variance_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

top_k_indices = np.argsort(all_uncertainties)[-5:]
montage_images = []

print("\n正在生成 Top-5 高不确定性样本可视化...")
for i, local_idx in enumerate(top_k_indices):
    dataset_idx = sample_indices[local_idx]
    image_tensor, _, vis_image, _ = valid_dataset[dataset_idx]
    
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets_for_cam)[0, :]
    
    rgb_img = vis_image.permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    entropy_val = all_uncertainties[local_idx]
    text_info = f"Top {i+1} H:{entropy_val:.3f}"
    
    img_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr, (0, 0), (224, 30), (0, 0, 0), -1)
    cv2.putText(img_bgr, text_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    img_bgr = cv2.copyMakeBorder(img_bgr, 0, 0, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    montage_images.append(img_bgr)
    cv2.imwrite(f"task2_uncertain_{i+1}.png", img_bgr)

if montage_images:
    summary_img = cv2.hconcat(montage_images)
    cv2.imwrite(f"task2_montage_{TARGET_CLASS_NAME}.png", summary_img)
    print("Top-5 汇总大图已保存。")
