import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 15
LR = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# =========================
# TRANSFORMS (Data Augmentation)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# =========================
# LOAD DATASET
# =========================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("📂 Classes:", train_dataset.classes)

# =========================
# MODEL (EfficientNet-B0)
# =========================
model = models.efficientnet_b0(weights="DEFAULT")

num_classes = len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(DEVICE)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN FUNCTION
# =========================
def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    acc = correct / len(train_dataset)
    return running_loss, acc

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate():
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()

    acc = correct / len(test_dataset)
    return acc

# =========================
# TRAIN LOOP
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    loss, train_acc = train_one_epoch()
    test_acc = evaluate()

    print(f"\n📌 Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_currency_model.pth")
        print("✅ Best model saved!")

# =========================
# SAVE FINAL MODEL
# =========================
torch.save(model.state_dict(), "currency_model.pth")

# Save class names
with open("classes.txt", "w") as f:
    for cls in train_dataset.classes:
        f.write(cls + "\n")

print("\n🎉 Training Completed!")
print(f"🏆 Best Accuracy: {best_acc:.4f}")
print("📦 Model saved as currency_model.pth")