import os
import shutil
import argparse
import random
from glob import glob

def split_dataset(img_dir, label_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    # Create output folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Collect image paths
    image_paths = glob(os.path.join(img_dir, '*.tif'))
    random.shuffle(image_paths)

    split_idx = int(len(image_paths) * (1 - val_ratio))
    train_images = image_paths[:split_idx]
    val_images = image_paths[split_idx:]

    print(f"🧪 Total images: {len(image_paths)} | Train: {len(train_images)} | Val: {len(val_images)}")

    # Helper to copy image and label
    def copy_files(image_list, split):
        for img_path in image_list:
            fname = os.path.basename(img_path)
            label_name = fname.replace('.tif', '.txt')
            label_path = os.path.join(label_dir, label_name)

            dest_img = os.path.join(output_dir, 'images', split, fname)
            dest_lbl = os.path.join(output_dir, 'labels', split, label_name)

            shutil.copyfile(img_path, dest_img)
            if os.path.exists(label_path):
                shutil.copyfile(label_path, dest_lbl)
            else:
                print(f"⚠️ Missing label for {fname}, skipping label.")

    copy_files(train_images, 'train')
    copy_files(val_images, 'val')

    print("✅ Dataset split completed.")

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory with .tif images")
    parser.add_argument('--label_dir', type=str, required=True, help="Directory with YOLO .txt labels")
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory (will create images/, labels/)")
    parser.add_argument('--val_ratio', type=float, default=0.2, help="Fraction of data to use as validation set")
    args = parser.parse_args()

    split_dataset(args.img_dir, args.label_dir, args.output_dir, args.val_ratio)
    """

    #xview dataset_tar ; tar files which are downloaded from xview website
    image_dir= "path/to/xViewDataset_tar/train/train_images
    label_dir = "path/to/xViewDataset_tar/labels/yolo_format_labels" # first change geojson to yolo format
    output_dir= None
    split_dataset(image_dir , label_dir , output_dir)
