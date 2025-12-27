import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'CheXpert-v1.0-small'
TRAIN_CSV = os.path.join(DATA_DIR, 'CheXpert-v1.0-small', 'train.csv')
VALID_CSV = os.path.join(DATA_DIR, 'CheXpert-v1.0-small', 'valid.csv')

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

CORE_CLASSES = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
]

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, data_dir, is_train=True):
        self.df = pd.read_csv(csv_file)
        
        for col in CORE_CLASSES:
            if col not in self.df.columns:
                self.df[col] = 0
        self.df = self.df.dropna(subset=CORE_CLASSES)
        self.df[CORE_CLASSES] = self.df[CORE_CLASSES].fillna(0).replace(-1, 0)
        
        self.data_dir = data_dir
        self.raw_paths = []
        
        for p in self.df['Path']:
            full_path = os.path.join(self.data_dir, p)
            self.raw_paths.append(full_path)
            
    def __len__(self):
        return len(self.df)

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
        return image_tensor, torch.from_numpy(labels)

def main():
    print(f"Device: {DEVICE}")
    
    try:
        train_dataset = CheXpertDataset(TRAIN_CSV, DATA_DIR, is_train=True)
        valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print(f"训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}")
    except Exception as e:
        print(f"数据加载错误: {e}")
        return

    print("加载 XRV 预训练权重...")
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    
    model.op_threshs = None

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(CORE_CLASSES)) 
    
    model = model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("开始微调...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} 完成. 平均 Loss: {avg_loss:.4f}")
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        try:
            aucs = [roc_auc_score(all_labels[:, j], all_preds[:, j]) for j in range(len(CORE_CLASSES))]
            print(f"Epoch {epoch+1} 验证集平均 AUC: {np.mean(aucs):.4f}")
            print(f"  详细 AUC: {dict(zip(CORE_CLASSES, np.round(aucs, 3)))}")
        except:
            print("AUC 计算失败 (可能是某个类别样本不足)")

    save_path = "chexpert_finetuned_xrv.pth"
    torch.save(model.state_dict(), save_path)
    print(f"微调完成！模型已保存至: {save_path}")

if __name__ == "__main__":
    main()
