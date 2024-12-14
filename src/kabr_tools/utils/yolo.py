import numpy as np
from ultralytics import YOLO


class YOLOv8:
    def __init__(self, weights="yolov8x.pt", imgsz=640, conf=0.5):
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(weights)
        self.names = self.model.names

    def forward(self, image):
        width = image.shape[1]
        height = image.shape[0]
        boxes = self.model.predict(source=image[:, :, ::-1].astype(np.uint8),
                                   imgsz=self.imgsz, iou=0.5, nms=True, agnostic_nms=True, verbose=False)[0].boxes.cpu()
        filtered = []

        for box, label, confidence in zip(boxes.xyxyn.numpy(), boxes.cls.numpy(), boxes.conf.numpy()):
            if confidence > self.conf:
                if self.names[label] in ["zebra", "horse", "giraffe"]:
                    box[0] = int(box[0] * width)
                    box[1] = int(box[1] * height)
                    box[2] = int(box[2] * width)
                    box[3] = int(box[3] * height)
                    box = box.astype(np.int32)
                    confidence = float(f"{confidence:.2f}")

                    if self.names[label] == "horse":
                        label = "Zebra"
                    else:
                        label = self.names[label].capitalize()

                    filtered.append(([box[0], box[1], box[2], box[3]], confidence, label))

        return filtered

    @staticmethod
    def get_centroid(box):
        x = int(box[0] + (box[2] - box[0]) / 2)
        y = int(box[1] + (box[3] - box[1]) / 2)

        return x, y
