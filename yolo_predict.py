from ultralytics import YOLO

model = YOLO(
    "best/weights/from/runs/"
)
image_path = None # path of image on which you want to predict
model.predict(source=image_path, save=True)
