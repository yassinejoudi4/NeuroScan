import os
import shutil
import random

DATA_DIR = "data"
DEST_DIR = "data"
SPLIT = (0.8, 0.1, 0.1)
SEED = 42

RENAME_MAP = {
    "1": "meningioma",
    "2": "glioma",
    "3": "pituitary",
}

random.seed(SEED)

for old_name, new_name in RENAME_MAP.items():
    old_path = os.path.join(DATA_DIR, old_name)
    new_path = os.path.join(DATA_DIR, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

if os.path.exists(os.path.join(DEST_DIR, "train")):
    print("Split déjà existant, arrêt.")
    exit()

classes = list(RENAME_MAP.values())

for cls in classes:
    src_path = os.path.join(DATA_DIR, cls)
    images = os.listdir(src_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, files in split_map.items():
        out_dir = os.path.join(DEST_DIR, split_name, cls)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            shutil.move(os.path.join(src_path, f), os.path.join(out_dir, f))

    print(f"{cls}: {n} images -> train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}")

for cls in classes:
    src_path = os.path.join(DATA_DIR, cls)
    if os.path.exists(src_path) and not os.listdir(src_path):
        os.rmdir(src_path)