from ultralytics import YOLO

model = YOLO(
    "/home/arjunsingh/omnipresent_projects/objectDetection2/runs/train/exp_xview15/weights/best.pt"
)
image_path = "/c/Users/91704/.cache/kagglehub/datasets/mdrifaturrahman33/levir-cd/versions/1/LEVIR CD/train/B/train_7.png"
model.predict(source=image_path, save=True)
