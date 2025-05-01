'''
This script collects images from the FER and RAF datasets, categorizes them into stress labels,
shuffles them, and splits them into training, validation, and test sets. The images are then saved in a structured directory format for further use, such as training a GAN model.
There is a 70/10/20 split for training, validation, and testing respectively.

To run this script, place FER-2013 in fer-dataset/ and RAF-DB in raf-dataset/
Make sure to adjust the BASE_DIR variable to point to your project folder.
'''
import os
from pathlib import Path
from PIL import Image
import random
import shutil

# === CONFIG: SET TO YOUR PROJECT FOLDER ===
BASE_DIR = Path("C:/Users/travi/Code/Stress-anxiety")
RAF_DIR = BASE_DIR / "raf-dataset" / "DATASET"
FER_DIR = BASE_DIR / "fer-dataset"
OUTPUT_DIR = BASE_DIR / "output-stress-split"
RANDOM_SEED = 42

# === Mappings ===
raf_folder_to_emotion = {
    "1": "surprise", "2": "fear", "3": "disgust",
    "4": "happy", "5": "sad", "6": "angry", "7": "neutral"
}
emotion_to_stress = {
    "neutral": "relaxed",
    "happy": "relaxed",
    "surprise": "mild_stress",
    "fear": "anxious",
    "sad": "anxious",
    "disgust": "ptsd",
    "angry": "ptsd"
}

# Collect images from RAF dataset
def collect_raf_images(raf_root):
    print(f"Scanning RAF-DB at: {raf_root}")
    stress_data = {k: [] for k in emotion_to_stress.values()}
    count = 0
    for split in ["train", "test"]:
        for folder in (raf_root / split).iterdir():
            emotion = raf_folder_to_emotion.get(folder.name)
            if not emotion:
                print(f"Skipped unknown RAF folder: {folder.name}")
                continue
            stress = emotion_to_stress.get(emotion)
            for img_path in folder.glob("*.jpg"):
                stress_data[stress].append(img_path)
                count += 1
    print(f"RAF-DB: Collected {count} images\n")
    return stress_data

# Collect images from FER dataset
def collect_fer_images(fer_root):
    print(f"Scanning FER-2013 at: {fer_root}")
    stress_data = {k: [] for k in emotion_to_stress.values()}
    count = 0
    for split in ["train", "test"]:
        for emotion_folder in (fer_root / split).iterdir():
            emotion = emotion_folder.name.lower()
            stress = emotion_to_stress.get(emotion)
            if not stress:
                print(f"Skipped unknown FER folder: {emotion_folder.name}")
                continue
            for img_path in emotion_folder.glob("*"):
                if img_path.is_file():
                    stress_data[stress].append(img_path)
                    count += 1
    print(f"FER: Collected {count} images\n")
    return stress_data

# Combine and split datasets
def merge_and_split(data_dicts, output_dir):
    print(f"Saving to: {output_dir.resolve()}")
    random.seed(RANDOM_SEED)
    combined = {k: [] for k in emotion_to_stress.values()}

    for data in data_dicts:
        for label, imgs in data.items():
            combined[label].extend(imgs)

    for label, all_paths in combined.items():
        if not all_paths:
            print(f"No data for class: {label}")
            continue
        random.shuffle(all_paths)
        n = len(all_paths)
        n_train = int(0.7 * n)
        n_val = int(0.1 * n)
        train, val, test = all_paths[:n_train], all_paths[n_train:n_train + n_val], all_paths[n_train + n_val:]

        for split, items in [("train", train), ("val", val), ("test", test)]:
            out_dir = output_dir / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in items:
                save_grayscale(img_path, out_dir)
            print(f"{split}/{label}: {len(items)} images")

# Grayscale save function
def save_grayscale(src_path, dst_dir):
    try:
        img = Image.open(src_path).convert("L")
        img.save(dst_dir / src_path.name)
    except Exception as e:
        print(f"Failed on {src_path.name}: {e}")

# Run Pipeline
print("=== CONVERTING RAF + FER TO STRESS CLASSES ===")

raf_data = collect_raf_images(RAF_DIR)
fer_data = collect_fer_images(FER_DIR)
merge_and_split([raf_data, fer_data], OUTPUT_DIR)

print("All done. Combined dataset saved to:", OUTPUT_DIR.resolve())
