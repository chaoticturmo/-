import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import numpy as np

DATA_DIR = 'CheXpert-v1.0-small'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
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

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    train_dataset = CheXpertDataset(TRAIN_CSV, DATA_DIR, transform=data_transform)
    valid_dataset = CheXpertDataset(VALID_CSV, DATA_DIR, transform=data_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"成功加载 {len(train_dataset)} 个训练样本和 {len(valid_dataset)} 个验证样本。")

except FileNotFoundError:
    print(f"错误: 无法在 '{DATA_DIR}' 找到 'train.csv' 或 'valid.csv'")
    print("请确保 DATA_DIR 路径正确。")
    exit()


model = models.densenet121(pretrained=True)

num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("--- 开始训练 ---")

for epoch in range(NUM_EPOCHS):
    model.train() 
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} 训练完成, 平均 Loss: {running_loss / len(train_loader):.4f}")

    model.eval() 
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            preds = torch.sigmoid(outputs)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        except ValueError:
            pass
            
    mean_auc = np.mean(auc_scores)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], **验证集平均 AUC: {mean_auc:.4f}**")

print("--- 训练完成 ---")

MODEL_SAVE_PATH = 'chexpert_model.pth'
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"模型已保存到 {MODEL_SAVE_PATH}")

