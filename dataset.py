import os

BASE_DIR = "dataset"

mapping = {
    "Tennote": "INR_10",
    "Twentynote": "INR_20",
    "Fiftynote": "INR_50",
    "1Hundrednote": "INR_100",
    "2Hundrednote": "INR_200",
    "5Hundrednote": "INR_500",
    "2Thousandnote": "INR_2000"
}

for split in ["Train", "Test"]:   # your folder names
    split_path = os.path.join(BASE_DIR, split)

    for old_name, new_name in mapping.items():
        old_path = os.path.join(split_path, old_name)
        new_path = os.path.join(split_path, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"✅ {old_name} → {new_name} ({split})")

print("🎉 Renaming completed!")