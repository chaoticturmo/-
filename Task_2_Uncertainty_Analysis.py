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
import cv2 


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = 'CheXpert-v1.0-small'
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')
MODEL_PATH = 'chexpert_model.pth'

BATCH_SIZE = 32 
CLASSES = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
]
NUM_CLASSES = len(CLASSES)


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file).dropna(subset=CLASSES)
        self.data_dir = data_dir
        self.transform = transform
        self.df[CLASSES] = self.df[CLASSES].fillna(0).replace(-1, 0)
        
   
        self.raw_paths = self.df['Path'].apply(lambda x: os.path.join(self.data_dir, x)).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.raw_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        
        vis_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        vis_image = vis_transform(image)
        
        labels = self.df.iloc[idx][CLASSES].values.astype(np.float32)
        return image_tensor, torch.from_numpy(labels), vis_image, idx

def load_model(path, num_classes):
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    map_loc = torch.device('cpu') if not torch.cuda.is_available() else None
    model.load_state_dict(torch.load(path, map_location=map_loc))
    return model

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR, transform=data_transform)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = load_model(MODEL_PATH, NUM_CLASSES).to(DEVICE).eval()


target_layers = [model.features.norm5] 
cam = GradCAM(model=model, target_layers=target_layers)

print("开始计算不确定性 (熵) 和解释分散性 (热力图方差)...")

all_uncertainties = []
all_dispersions = []
sample_indices = [] 


TARGET_CLASS_IDX = 0 
targets_for_cam = [ClassifierOutputTarget(TARGET_CLASS_IDX)]

with torch.no_grad():
    for (images, _, _, batch_indices) in valid_loader:
        images = images.to(DEVICE)
        

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        

        p = probs
        epsilon = 1e-9
        binary_entropy = - (p * torch.log2(p + epsilon) + (1 - p) * torch.log2(1 - p + epsilon))
        
        sample_uncertainty = torch.mean(binary_entropy, dim=1) # (N,)
        all_uncertainties.extend(sample_uncertainty.cpu().numpy())
        
        torch.set_grad_enabled(True)
        grayscale_cam = cam(input_tensor=images, targets=targets_for_cam)
        torch.set_grad_enabled(False)
        

        dispersions = np.var(grayscale_cam, axis=(1, 2))
        all_dispersions.extend(dispersions)
        
        sample_indices.extend(batch_indices.numpy())
        
        if len(sample_indices) % 100 == 0:
            print(f"  已处理 {len(sample_indices)} / {len(valid_dataset)} 样本...")

print("计算完成。")
 
all_uncertainties = np.array(all_uncertainties)
all_dispersions = np.array(all_dispersions)


mask = ~np.isnan(all_uncertainties) & ~np.isnan(all_dispersions)
uncertainties_clean = all_uncertainties[mask]
dispersions_clean = all_dispersions[mask]

corr_p, p_val_p = pearsonr(uncertainties_clean, dispersions_clean)
corr_s, p_val_s = spearmanr(uncertainties_clean, dispersions_clean)

print("\n--- 任务2 交付物: 不确定性与解释分散性的相关性  ---")
print(f"不确定性 (熵) vs. 解释分散性 (热力图方差) for class '{CLASSES[TARGET_CLASS_IDX]}'")
print(f"  皮尔逊相关系数 (Pearson): {corr_p:.4f} (p-value: {p_val_p:.4f})")
print(f"  斯皮尔曼相关系数 (Spearman): {corr_s:.4f} (p-value: {p_val_s:.4f})")
print("----------------------------------------------------------")


print("\n正在生成高不确定性样本的可视化...")


top_uncertain_indices = np.argsort(all_uncertainties)[-5:]


model.eval()
torch.set_grad_enabled(True)

for i, sample_idx in enumerate(top_uncertain_indices):

    image_tensor, _, vis_image, _ = valid_dataset[sample_idx]
    
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets_for_cam)[0, :] # (H, W)
    
    rgb_img = vis_image.permute(1, 2, 0).numpy()
    
    
    if rgb_img.max() <= 1.0:
        rgb_img = (rgb_img * 255).astype(np.uint8)
        
    
    visualization = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
    
    save_path = f"task2_uncertain_sample_{i+1}.png"
    cv2.imwrite(save_path, visualization)
    print(f"  已保存高不确定性样本 {i+1} (index {sample_idx}) 到 {save_path}")

print("\n任务2 分析完成。")