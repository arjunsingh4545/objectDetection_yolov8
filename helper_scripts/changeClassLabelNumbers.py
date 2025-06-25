import os
from glob import glob

# === Config ===
label_dir = "/part/data_fat32/xViewDataset_tar/labels/yolo_format_labels_copy"

# === Class ID mapping (original → new YOLOv8-friendly 0-based index) ===
id_map = {
    11: 0, 12: 1, 13: 2, 15: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 23: 9,
    24: 10, 25: 11, 26: 12, 27: 13, 28: 14, 29: 15, 32: 16, 33: 17, 34: 18, 35: 19,
    36: 20, 37: 21, 38: 22, 40: 23, 41: 24, 42: 25, 44: 26, 45: 27, 47: 28, 49: 29,
    50: 30, 51: 31, 52: 32, 53: 33, 54: 34, 55: 35, 56: 36, 57: 37, 59: 38, 60: 39,
    61: 40, 62: 41, 63: 42, 64: 43, 65: 44, 66: 45, 71: 46, 72: 47, 73: 48, 74: 49,
    75: 50, 76: 51, 77: 52, 79: 53, 82: 54, 83: 55, 84: 56, 86: 57, 89: 58, 91: 59,
    93: 60, 94: 61
}

# === Process all .txt files ===
txt_files = glob(os.path.join(label_dir, "*.txt"))
print(f"📝 Processing {len(txt_files)} label files...")

for txt_file in txt_files:
    new_lines = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_id = int(float(parts[0]))  # Handles 15.0 etc
            except ValueError:
                print(f"⚠️ Invalid line skipped in {txt_file}: {line.strip()}")
                continue
            if old_id not in id_map:
                print(f"⚠️ Warning: Class ID {old_id} not in mapping. Skipping line in {txt_file}")
                continue
            parts[0] = str(id_map[old_id])
            new_lines.append(" ".join(parts))

    # Write the updated lines back to the file
    with open(txt_file, 'w') as f:
        f.write("\n".join(new_lines) + "\n")

print("✅ Class ID remapping completed.")

