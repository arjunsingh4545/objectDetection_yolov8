import argparse
from ultralytics import YOLO

"""
def train_yolo(model_path, data_path, epochs, imgsz, batch, device):
    print("🚀 Starting YOLOv8 training...")

    model = YOLO(model_path)
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        verbose=True,
        project='runs/train',
        name='exp_xview',

        # augmentations
        degrees=10,         # random rotation (-10 to +10 degrees)
        translate=0.1,      # shift up/down/left/right (10%)
        scale=0.5,          # zoom in/out up to 50%
        shear=2.0,          # shear image
        perspective=0.0005, # apply perspective transform
        flipud=0.5,         # vertical flip probability
        fliplr=0.5,         # horizontal flip probability
        mosaic=1.0,         # enable mosaic augmentation
        mixup=0.2,          # mixup augmentation probability
        hsv_h=0.015,        # hue variation
        hsv_s=0.7,          # saturation variation
        hsv_v=0.4,
    )

    print("✅ Training complete!")
"""


def train_yolo_light_aug(model_path, data_path, epochs, imgsz, batch, device):
    import os
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    print("🚀 Starting YOLOv8 training...")
    model = YOLO(model_path)

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,  # Reduce if still OOM
        project="runs/train",
        name="exp_xview",
        verbose=True,
        # learning rate
        lr0=1e-2,
        lrf=1e-5,
        cos_lr=True,
        # class loss
        # cls=0.7,
        patience=10,
        # Lighter augmentations
        degrees=10.0,
        scale=0.2,
        flipud=0.5,
        fliplr=0.5,
        hsv_v=0.4,
        hsv_h=0.015,
        mosaic=1.0,
    )

    print("✅ Training complete!")


def train_yolo_2(model_path, data_path, epochs, imgsz, batch, device):
    import os
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    print("🚀 Starting YOLOv8 training...")
    model = YOLO(model_path)

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,  # Reduce if still OOM
        project="runs/train",
        name="exp_xview",
        verbose=True,
        # learning rate
        lr0=1e-2,
        lrf=1e-5,
        cos_lr=True,
        # loss
        cls=0.3,
        box=0.2,
        dfl=1.5,
        patience=10,
        # Lighter augmentations
        degrees=5.0,
        scale=0.1,
        flipud=0.3,
        fliplr=0.5,
        hsv_v=0.4,
        hsv_h=0.015,
        mosaic=0.2,
        mixup=0.05,
        translate=0.05,
    )
    print("✅ Training complete!")


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help="Path to YOLOv8 model or pretrained weight")
    parser.add_argument('--data', type=str, required=True, help="Path to data.yaml file")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size (square)")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--device', type=str, default='0', help="Device: 0 (GPU), 'cpu', or '0,1' for multi-GPU")
    args = parser.parse_args()

    train_yolo(args.model, args.data, args.epochs, args.imgsz, args.batch, args.device)
    """
    """
    model_path = "yolov8m.pt"
    data_path = f"/part/data_fat32/xView_yolo_tiles/data.yaml"
    epochs = 25 
    imgsz = 640
    batch = 3
    device = '0'
    train_yolo_light_aug(model_path , data_path , epochs ,imgsz , batch , device)
    """
    model_path = "yolov8s.pt"
    data_path = f"/part/data_fat32/xView_yolo_tiles/data.yaml"
    epochs = 100
    imgsz = 768
    batch = 3
    device = "0"
    train_yolo_2(model_path, data_path, epochs, imgsz, batch, device)
