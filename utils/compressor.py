import os
import sys
import cv2
from natsort import natsorted
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python compressor.py path_to_image path_to_video annotation")
        exit(0)
    elif len(sys.argv) == 3:
        path_to_image = sys.argv[1]
        path_to_video = sys.argv[2]
        annotation = None
    elif len(sys.argv) == 4:
        path_to_image = sys.argv[1]
        path_to_video = sys.argv[2]
        annotation = sys.argv[3]

    if not os.path.exists(path_to_video):
        os.makedirs(path_to_video)

    folders = natsorted(os.listdir(path_to_image))

    label2number = {"Walk": 0,
                    "Graze": 1,
                    "Browse": 2,
                    "Head Up": 3,
                    "Auto-Groom": 4,
                    "Trot": 5,
                    "Run": 6,
                    "Occluded": 7}

    number2label = {value: key for key, value in label2number.items()}

    if annotation is not None:
        df = pd.read_csv(annotation, sep=" ")

    for i, folder in enumerate(folders):
        print(f"{i + 1}/{len(folders)}")

        if annotation is not None:
            mapping = {}

            for index, row in df[df.original_vido_id == folder].iterrows():
                mapping[row["frame_id"]] = number2label[row["labels"]]

        vw = cv2.VideoWriter(f"{path_to_video}/{folder}.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"), 29.97,
                             (400, 300))

        for j, file in enumerate(natsorted(os.listdir(path_to_image + os.sep + folder))):
            image = cv2.imread(f"{path_to_image}/{folder}/{file}")

            if annotation is not None:
                color = (0, 0, 0)
                label = mapping[j + 1]
                thickness_in = 1
                size = 0.7
                label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)
                copied = image.copy()
                cv2.rectangle(image, (10, 10), (20 + label_length[0][0], 40), (255, 255, 255), -1)
                cv2.putText(image, label, (16, 31),
                            cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i - 50 for i in color]), thickness_in, cv2.LINE_AA)
                image = cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)

            vw.write(image)

        vw.release()
