# ===============================
# Practical 6: Transfer Learning (VGG16) — Colab Ready Version
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time, copy, os

# ------------------------------
# 1. Image transformations
# ------------------------------
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ------------------------------
# 2. Download CIFAR-10 dataset
# ------------------------------
data_dir = "/content/data"
os.makedirs(data_dir, exist_ok=True)

train_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=image_transforms['train'])
valid_data = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=image_transforms['valid'])

batch_size = 32
dataloaders = {
    'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
    'valid': DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
}

class_names = train_data.classes
n_classes = len(class_names)
print("✅ CIFAR-10 loaded successfully!")
print("Classes:", class_names)

# ------------------------------
# 3. Load pretrained VGG16
# ------------------------------
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
print("\n✅ Pretrained VGG16 loaded")

# Freeze convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier head
n_inputs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, n_classes),
    nn.LogSoftmax(dim=1)
)
print("\nModified classifier head:\n", model.classifier)

# ------------------------------
# 4. Define loss and optimizer
# ------------------------------
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ------------------------------
# 5. Device setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"\n💻 Using device: {device}")

# ------------------------------
# 6. Training loop
# ------------------------------
n_epochs = 3
train_loss_history, valid_loss_history = [], []

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    print("-" * 25)

    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss, running_corrects = 0.0, 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        if phase == 'train':
            train_loss_history.append(epoch_loss)
        else:
            valid_loss_history.append(epoch_loss)

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    # Free GPU memory
    torch.cuda.empty_cache()

# ------------------------------
# 7. Plot training curves
# ------------------------------
plt.figure(figsize=(8, 6))
plt.plot(train_loss_history, label='Training Loss', marker='o')
plt.plot(valid_loss_history, label='Validation Loss', marker='s')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 8. Save trained model
# ------------------------------
torch.save(model.state_dict(), '/content/transfer_learning_vgg16.pth')
print("\n✅ Model saved to /content/transfer_learning_vgg16.pth")
