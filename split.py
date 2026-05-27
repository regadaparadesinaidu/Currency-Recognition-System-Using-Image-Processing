import os
import shutil
import random

# ===== PATHS =====
SOURCE_DIR = "us_data/USA currency"
TARGET_DIR = "dataset"

TRAIN_DIR = os.path.join(TARGET_DIR, "train")
TEST_DIR = os.path.join(TARGET_DIR, "test")

SPLIT_RATIO = 0.8

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print("🚀 Processing US dataset...\n")

for class_name in os.listdir(SOURCE_DIR):

    # ❌ Skip unwanted folders
    if class_name.lower() == "test set":
        continue

    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    # ✅ Extract number from "1 Dollar"
    value = class_name.split()[0]   # "1 Dollar" → "1"
    new_class_name = f"USD_{value}"

    print(f"{class_name} → {new_class_name}")

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    train_class_dir = os.path.join(TRAIN_DIR, new_class_name)
    test_class_dir = os.path.join(TEST_DIR, new_class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy train
    for img in train_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Copy test
    for img in test_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(f"✅ Done {new_class_name}: {len(train_imgs)} train, {len(test_imgs)} test\n")

print("🎉 US dataset ready!")