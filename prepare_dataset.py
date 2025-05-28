import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import random
import os
import pickle

# Config
output_dir = './data_splits'
os.makedirs(output_dir, exist_ok=True)
random.seed(42)

# Load CIFAR-100
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(256, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.CIFAR100(root="./full_dataset", train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR100(root="./full_dataset", train=False, download=False, transform=transform_test)

# Select Classes
class_names = ['bicycle', 'bus', 'lawn_mower', 'motorcycle', 'pickup_truck', 'rocket', 'streetcar', 'tank', 'tractor', 'train']

selected_classes = []
for i in range(len(train_dataset.classes)):
    if train_dataset.classes[i] in class_names:
        selected_classes.append(i)

print("Selected classes:", class_names)

# Filter Dataset
class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, selected_classes):
        self.data = []
        for img, label in dataset:
            if label in selected_classes:
                new_label = selected_classes.index(label)
                self.data.append((img, new_label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

train_filtered = FilteredDataset(train_dataset, selected_classes)
test_filtered_full = FilteredDataset(test_dataset, selected_classes) 

# Split Validation from Test
val_ratio = 0.5  # % for validation
total = len(test_filtered_full)
indices = list(range(total))
random.shuffle(indices)
val_split = int(total * val_ratio)

val_indices = indices[:val_split]
test_indices = indices[val_split:]

val_filtered = torch.utils.data.Subset(test_filtered_full, val_indices)
test_filtered = torch.utils.data.Subset(test_filtered_full, test_indices)

def save_filtered_dataset(dataset, filename):
    images = torch.stack([img for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])
    torch.save({'images': images, 'labels': labels}, os.path.join(output_dir, filename))

# Save Datasets
save_filtered_dataset(train_filtered, 'train_filtered.pt')
save_filtered_dataset(val_filtered, 'val_filtered.pt')
save_filtered_dataset(test_filtered, 'test_filtered.pt')

# Save selected classes
with open(os.path.join(output_dir, 'selected_classes.pkl'), 'wb') as f:
    pickle.dump(class_names, f)

print(f" Saved filtered train/val/test datasets and class names in {output_dir}")
