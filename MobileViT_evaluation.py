import json
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import pickle
import timm
import numpy as np
import seaborn as sns

# Config
LOG_PATH = './logs/mobilevit_xxs_training_log_newest.json'
MODEL_PATH = './models/mobilevit_xxs_newest.pth'
MODEL_NAME = 'mobilevit_xxs'
BATCH_SIZE = 64

#LOG_PATH = './logs/mobilevit_xs_training_log_92point6.json'
#MODEL_PATH = './models/mobilevit_xs_92point6.pth'
#MODEL_NAME = 'mobilevit_xs'
#BATCH_SIZE = 32

#LOG_PATH = './logs/mobilevit_s_training_log_93point6.json'
#MODEL_PATH = './models/mobilevit_s_93point6.pth'
#MODEL_NAME = 'mobilevit_s'
#BATCH_SIZE = 16

DATA_PATH = './data_splits'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (3, 256, 256)
OUTPUT_PATH = "./plots"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Data
class_names = None
with open(os.path.join(DATA_PATH, 'selected_classes.pkl'), 'rb') as f:
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
    
test_data = torch.load(os.path.join(DATA_PATH, 'test_filtered.pt'))
test_dataset = SimpleDataset(test_data['images'], test_data['labels'])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH)
model.to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)

# Training plots 
def load_training_logs(log_path):
    with open(log_path, 'r') as f:
        raw_logs = json.load(f)

    logs = {
        "epochs": [entry["epoch"] for entry in raw_logs],
        "train_loss": [entry["train_loss"] for entry in raw_logs],
        "val_loss": [entry["val_loss"] for entry in raw_logs],
        "train_acc": [entry["train_acc"] for entry in raw_logs],
        "val_acc": [entry["val_acc"] for entry in raw_logs],
    }
    return logs

def plot_accuracy_loss(logs):
    epochs = logs['epochs']

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, logs['train_loss'], label='Train Loss')
    plt.plot(epochs, logs['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, logs['train_acc'], label='Train Accuracy')
    plt.plot(epochs, logs['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH,f'{MODEL_NAME}_accuracy_loss_vs_epochs_new.png'))
    plt.close()


# Confusion matrix
def generate_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)

def print_confusion_matrix(cm, class_names):
    print("Confusion Matrix (rows = true labels, columns = predicted labels):")
    print("\t" + "\t".join(class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]}\t" + "\t".join(map(str, row)))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH,f'{MODEL_NAME}_confusion_matrix_new.png'))
    plt.close()

def calculate_metrics(cm):
    epsilon = 1e-10
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    fp = np.sum(cm, axis=0) - tp
    tn = np.sum(cm) - (tp + fp + fn)
    
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")

    accuracy = np.sum(tp) / np.sum(cm)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    return accuracy, precision, recall, specificity, f1

def evaluate_model_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

live_accuracy = evaluate_model_accuracy(model, test_loader, DEVICE)
print(f"Direct Accuracy (live): {live_accuracy:.4f}")


# Training plots
logs = load_training_logs(LOG_PATH)
plot_accuracy_loss(logs)

# Confusion matrix
cm = generate_confusion_matrix(model, test_loader, DEVICE)
print_confusion_matrix(cm, class_names)

accuracy, precision, recall, specificity, f1_score = calculate_metrics(cm)

print("\nMetrics Based on Confusion Matrix:")
print(f"Accuracy: {accuracy:.4f}")
for i, name in enumerate(class_names):
    print(f"\nClass: {name}")
    print(f"  Precision:  {precision[i]:.4f}")
    print(f"  Recall:     {recall[i]:.4f}")
    print(f"  Specificity:{specificity[i]:.4f}")
    print(f"  F1-score:   {f1_score[i]:.4f}")

print("Evaluate model accuracy", evaluate_model_accuracy(model,test_loader,DEVICE))
