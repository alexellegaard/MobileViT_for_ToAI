import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from thop import profile
import pickle
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import numpy as np
import json

# Best configuration so far
# LEARNING_RATE = 3e-4
# WEIGHT_DECAY = 5e-2
# drop_rate=0.2
# drop_path_rate=0.05
# label_smoothing=0.1
# scheduler=ReduceLROnPlateau(optimizer,mode='max', factor=0.25, patience=5, threshold=0.001)
# MODEL_NAME = 'mobilevit_xxs'

# Best configuration so far
# LEARNING_RATE = 3e-4
# WEIGHT_DECAY = 5e-2
# drop_rate=0.1
# drop_path_rate=0.05
# label_smoothing=0.05
# scheduler=ReduceLROnPlateau(optimizer,mode='max', factor=0.25, patience=5, threshold=0.001)
# MODEL_NAME = 'mobilevit_xs'

# Config 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-2
MODEL_NAME = 'mobilevit_xxs'

# Paths
dataset_dir = './data_splits'
model_dir = './models'
log_dir = './tensorboard_logs'
os.makedirs(model_dir, exist_ok=True)   
os.makedirs(log_dir, exist_ok=True)

# TensorBoard Writer
writer = SummaryWriter(log_dir=log_dir) 

# Data Augmentation
class_names = None
with open(os.path.join(dataset_dir, 'selected_classes.pkl'), 'rb') as f:
    class_names = pickle.load(f)

NUM_CLASSES = len(class_names)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_data = torch.load(os.path.join(dataset_dir, 'train_filtered.pt'))
train_dataset = SimpleDataset(train_data['images'], train_data['labels'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_data = torch.load(os.path.join(dataset_dir, 'val_filtered.pt'))
val_dataset = SimpleDataset(val_data['images'], val_data['labels'])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model 
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, drop_rate=0.2, drop_path_rate=0.1)
model = model.to(DEVICE)

#print(model.default_cfg['input_size'])  # Returns (3, 256, 256)

# # Model performance metrics
# macs, params = profile(model, inputs=(torch.randn(1, 3, 256, 256).to(DEVICE),))
# print(f"FLOPs: {macs/1e6:.2f} MFLOPs, Params: {params/1e6:.2f} M")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min = 1e-6)
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.25, patience=5, threshold=0.001)

# Training and Validation
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    # F1 and recall
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    recall = recall_score(all_labels, all_preds, average='macro') * 100

    # Specificity = TN / (TN + FP), macro-averaged
    cm = confusion_matrix(all_labels, all_preds)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class) * 100

    return epoch_loss, epoch_acc, f1, recall, specificity

# Early stopping params
early_stop_patience = 20
no_improve_epochs = 0
best_acc = 0.0
best_epoch = 0

# Main loop
print(f"Executing training/validation loop on {DEVICE} hardware")
train_log = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc, f1, recall, specificity = validate()

    scheduler.step(val_acc)

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Accuracy/Validation', val_acc, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('Metrics/F1', f1, epoch)
    writer.add_scalar('Metrics/Recall', recall, epoch)
    writer.add_scalar('Metrics/Specificity', specificity, epoch)

    print(f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
        f"F1: {f1:.2f} | Recall: {recall:.2f} | Spec: {specificity:.2f}")


    # Check for improvement
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
        no_improve_epochs = 0
        #torch.save(model.state_dict(), os.path.join(model_dir, 'mobilevit_cifar100_best.pth'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, os.path.join(model_dir, 'f"{MODEL_NAME}_newest.pth'))
        print("New best model saved.")
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epoch(s).")

    # Early stopping trigger
    if no_improve_epochs >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}. Best val accuracy: {best_acc:.2f}% at epoch {best_epoch}.")
        break
    
    train_log.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'f1': f1,
        'recall': recall,
        'specificity': specificity,
        'lr': optimizer.param_groups[0]['lr']
    })


print(f"Training complete. Best val accuracy: {best_acc:.2f}%")
writer.close()

with open(os.path.join('logs', f'{MODEL_NAME}_training_log_newest.json'), 'w') as f:
    json.dump(train_log, f, indent=2)
