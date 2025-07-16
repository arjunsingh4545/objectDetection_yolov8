import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------- CONFIG ----------------------
CONFIG = {
    'tile_size': 1024,
    'overlap': 200,
    'max_boxes': 1000,
    'input_base_img': "/path/to/xViewDataset_objectDetection_yolo/images",
    'input_base_lbl': "/path/to/xViewDataset_objectDetection_yolo/labels",
    'output_base_img': "/path/to/xView_yolo_tiles/images",
    'output_base_lbl': "/path/to/xView_yolo_tiles/labels",
    'splits': ['train', 'val']
}
# ----------------------------------------------------

def yolo_to_bbox(line, img_w, img_h):
    """Convert YOLO format line to absolute pixel bounding box."""
    cls, xc, yc, bw, bh = map(float, line.strip().split())
    xc *= img_w
    yc *= img_h
    bw *= img_w
    bh *= img_h
    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2
    return [int(cls), x1, y1, x2, y2]

def bbox_to_yolo(cls, x1, y1, x2, y2, img_w, img_h):
    """Convert pixel bounding box to YOLO format."""
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def tile_image(img, boxes, basename, split, config):
    """Tile a single image and write YOLO annotations for each tile."""
    h, w = img.shape[:2]
    step = config['tile_size'] - config['overlap']

    for y in range(0, h - config['tile_size'] + 1, step):
        for x in range(0, w - config['tile_size'] + 1, step):
            tile = img[y:y + config['tile_size'], x:x + config['tile_size']]
            tile_boxes = []

            for cls, x1, y1, x2, y2 in boxes:
                ix1 = max(x, x1)
                iy1 = max(y, y1)
                ix2 = min(x + config['tile_size'], x2)
                iy2 = min(y + config['tile_size'], y2)

                if ix1 < ix2 and iy1 < iy2:
                    nx1 = ix1 - x
                    ny1 = iy1 - y
                    nx2 = ix2 - x
                    ny2 = iy2 - y
                    tile_boxes.append(bbox_to_yolo(cls, nx1, ny1, nx2, ny2, config['tile_size'], config['tile_size']))

            if 0 < len(tile_boxes) <= config['max_boxes']:
                tile_name = f"{basename}_{x}_{y}.jpg"
                tile_path_img = os.path.join(config['output_base_img'], split, tile_name)
                tile_path_lbl = os.path.join(config['output_base_lbl'], split, tile_name.replace(".jpg", ".txt"))

                cv2.imwrite(tile_path_img, tile)
                with open(tile_path_lbl, 'w') as f:
                    f.write('\n'.join(tile_boxes))

def process_split(split, config):
    """Process all images in a split (train/val)."""
    img_dir = os.path.join(config['input_base_img'], split)
    lbl_dir = os.path.join(config['input_base_lbl'], split)
    out_img_dir = os.path.join(config['output_base_img'], split)
    out_lbl_dir = os.path.join(config['output_base_lbl'], split)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

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

        tile_image(img, boxes, fname.rsplit('.', 1)[0], split, config)

def main():
    for split in CONFIG['splits']:
        process_split(split, CONFIG)

if __name__ == "__main__":
    main()
