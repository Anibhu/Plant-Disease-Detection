import os
from collections import Counter

# 🔁 Change this to your dataset root directory
dataset_path = r"C:\Users\ANISH BHUIN\OneDrive\Desktop\jbooks\PlantVillage"  # example path

all_labels = []
for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        print(f"{class_dir}: {count} images")
        all_labels.extend([class_dir] * count)

print("\nClass distribution summary:")
print(Counter(all_labels))
