"""
Dental Condition Classification from Intraoral Images
EfficientNet-B0 classifier for 6 dental conditions.
Test accuracy: 92.37%
"""
import os, copy, random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = 'dental_clean'   # adjust to your local dataset path
CLASSES = sorted(os.listdir(DATA_DIR))
NUM_CLASSES = len(CLASSES)
print(f"Classes: {CLASSES}")

# Build file lists
all_paths, all_labels = [], []
for idx, cls in enumerate(CLASSES):
    for f in os.listdir(os.path.join(DATA_DIR, cls)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            all_paths.append(os.path.join(DATA_DIR, cls, f))
            all_labels.append(idx)
all_paths = np.array(all_paths); all_labels = np.array(all_labels)

# Stratified 80 / 15 / 5 split
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=SEED)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.25, stratify=temp_labels, random_state=SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class DentalDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.transform(img), self.labels[i]

train_loader = DataLoader(DentalDataset(train_paths, train_labels, train_tf),
                          batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(DentalDataset(val_paths, val_labels, eval_tf),
                          batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(DentalDataset(test_paths, test_labels, eval_tf),
                          batch_size=32, shuffle=False, num_workers=2)

# Class weights to handle imbalance
cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_labels)
cw = torch.tensor(cw, dtype=torch.float).to(device)

# EfficientNet-B0 with pretrained weights
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# Training with early stopping
EPOCHS, PATIENCE = 15, 4
best_acc, best_w, no_imp = 0.0, copy.deepcopy(model.state_dict()), 0

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total

for epoch in range(EPOCHS):
    tr_l, tr_a = run_epoch(train_loader, True)
    vl_l, vl_a = run_epoch(val_loader, False)
    scheduler.step()
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss {tr_l:.4f} Acc {tr_a*100:.2f}% "
          f"| Val Loss {vl_l:.4f} Acc {vl_a*100:.2f}%")
    if vl_a > best_acc:
        best_acc, best_w, no_imp = vl_a, copy.deepcopy(model.state_dict()), 0
        torch.save(best_w, 'best_dental_model.pth')
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Final test evaluation
model.load_state_dict(best_w)
model.eval()
preds, labels_all = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())

print(f"\n🎯 Test accuracy: {np.mean(np.array(preds) == np.array(labels_all)) * 100:.2f}%")
print(classification_report(labels_all, preds, target_names=CLASSES))
