# 🛰️ Object Detection in Satellite Images using YOLOv8

This repository contains a complete pipeline for detecting objects in satellite images using the [xView dataset](https://challenge.xviewdataset.org/data-explore) (60 classes) and the YOLOv8 object detection model.

---

## 📦 Dataset

- **Dataset:** [xView Dataset](https://challenge.xviewdataset.org/data-explore)
- **Format:** GeoJSON bounding boxes
- **Classes:** 60 object categories (e.g., buildings, vehicles, infrastructure)

---

## 🧩 Project Structure

### 🔁 `convert_geojson_to_yolo.py`
Converts the xView dataset annotations from **GeoJSON** format to **YOLO format**.

- Input: GeoJSON files
- Output: YOLO `.txt` annotation files for each image

### 🪟 `make_tiles_yolo.py`
Splits large satellite images into smaller **tiles** for training.

> ⚠️ **Optional step.** Useful for limited GPU VRAM (e.g., 4GB). Tiling improves trainability without compromising performance.

### ✂️ `split_train_val.py`
Splits the tiled dataset into **training and validation** sets.

- Default split: `80% training / 20% validation`

### 🏋️ `train_yolo.py`
Trains the YOLOv8 model on the dataset.

- **Model used:** `yolov8s.pt` (recommended for low-VRAM GPUs)
- If you have a better GPU, use `yolov8l.pt` for improved performance.
- Weights and logs are saved in the `runs/` directory.

### 🔍 `yolo_predict.py`
Performs inference on test images using the trained model.

- Utilizes weights from `runs/train/exp*/weights/best.pt`

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
