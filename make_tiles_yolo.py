import os
import cv2
import numpy as np
from tqdm import tqdm

# CONFIG
tile_size = 1024
overlap = 200
max_boxes = 1000

input_base_img = f"/part/data_fat32/xViewDataset_objectDetection_yolo/images"
input_base_lbl = f"/part/data_fat32/xViewDataset_objectDetection_yolo/labels"

output_base_img = "/part/data_fat32/xView_yolo_tiles/images"
output_base_lbl = "/part/data_fat32/xView_yolo_tiles/labels"

splits = ['train', 'val']

def yolo_to_bbox(line, w, h):
    cls, xc, yc, bw, bh = map(float, line.strip().split())
    xc *= w
    yc *= h
    bw *= w
    bh *= h
    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2
    return [int(cls), x1, y1, x2, y2]

def bbox_to_yolo(cls, x1, y1, x2, y2, w, h):
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def tile_image(img, boxes, basename, split):
    h, w = img.shape[:2]
    step = tile_size - overlap

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = img[y:y + tile_size, x:x + tile_size]

            tile_boxes = []
            for cls, x1, y1, x2, y2 in boxes:
                ix1 = max(x, x1)
                iy1 = max(y, y1)
                ix2 = min(x + tile_size, x2)
                iy2 = min(y + tile_size, y2)

                if ix1 < ix2 and iy1 < iy2:
                    nx1 = ix1 - x
                    ny1 = iy1 - y
                    nx2 = ix2 - x
                    ny2 = iy2 - y
                    tile_boxes.append(bbox_to_yolo(cls, nx1, ny1, nx2, ny2, tile_size, tile_size))

            if 0 < len(tile_boxes) <= max_boxes:
                tile_name = f"{basename}_{x}_{y}.jpg"
                tile_path_img = os.path.join(output_base_img, split, tile_name)
                tile_path_lbl = os.path.join(output_base_lbl, split, tile_name.replace(".jpg", ".txt"))
                cv2.imwrite(tile_path_img, tile)
                with open(tile_path_lbl, 'w') as f:
                    f.write('\n'.join(tile_boxes))

# Process train and val
for split in splits:
    img_dir = os.path.join(input_base_img, split)
    lbl_dir = os.path.join(input_base_lbl, split)
    os.makedirs(os.path.join(output_base_img, split), exist_ok=True)
    os.makedirs(os.path.join(output_base_lbl, split), exist_ok=True)

    for fname in tqdm(os.listdir(img_dir), desc=f"Tiling {split}"):
        if not fname.lower().endswith(('.jpg', '.png', '.tif')):
            continue

        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, fname.rsplit('.', 1)[0] + ".txt")
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                boxes = [yolo_to_bbox(line, w, h) for line in f if line.strip()]

        tile_image(img, boxes, fname.rsplit('.', 1)[0], split)

