import os
import sys
import cv2
from tqdm import tqdm
from src.yolo import YOLOv8
from src.tracker import Tracker, Tracks
from src.object import Object
from src.draw import Draw

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python detector2cvat path_to_videos path_to_save")
        exit(0)
    else:
        path_to_videos = sys.argv[1]
        path_to_save = sys.argv[2]

    videos = []

    for root, dirs, files in os.walk(path_to_videos):
        for file in files:
            if os.path.splitext(file)[1] == ".mp4":
                folder = root.split("/")[-1]

                if folder.startswith("!") or file.startswith("!"):
                    continue

                videos.append(f"{root}/{file}")

    yolo = YOLOv8(weights="yolov8x.pt", imgsz=3840, conf=0.5)

    for i, video in enumerate(videos):
        try:
            name = os.path.splitext(video.split("/")[-1])[0]

            output_folder = path_to_save + os.sep + "/".join(os.path.splitext(video)[0].split("/")[-3:-1])
            output_path = f"{output_folder}/{name}.xml"
            print(f"{i + 1}/{len(videos)}: {video} -> {output_path}")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            vc = cv2.VideoCapture(video)
            size = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = cv2.VideoWriter(f"{output_folder}/{name}_demo.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                 29.97, (width, height))
            max_disappeared = 40
            tracker = Tracker(max_disappeared=max_disappeared, max_distance=300)
            tracks = Tracks(max_disappeared=max_disappeared, interpolation=True,
                            video_name=name, video_size=size, video_width=width, video_height=height)
            index = 0
            vc.set(cv2.CAP_PROP_POS_FRAMES, index)
            pbar = tqdm(total=size)

            while vc.isOpened():
                returned, frame = vc.read()

                if returned:
                    visualization = frame.copy()
                    predictions = yolo.forward(frame)
                    centroids = []
                    attributes = []

                    for prediction in predictions:
                        attribute = {}
                        centroids.append(YOLOv8.get_centroid(prediction[0]))
                        attribute["box"] = prediction[0]
                        attribute["confidence"] = prediction[1]
                        attribute["label"] = prediction[2]
                        attributes.append(attribute)

                    objects, colors = tracker.update(centroids)
                    objects = Object.object_factory(objects, centroids, colors, attributes=attributes)
                    tracks.update(objects, index)

                    for object in objects:
                        Draw.track(visualization, tracks[object.object_id].centroids, object.color, 20)
                        Draw.bounding_box(visualization, object)
                        Draw.object_id(visualization, object)

                    cv2.putText(visualization, f"Frame: {index}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.imshow("detector2cvat", cv2.resize(visualization, (int(width // 2.5), int(height // 2.5))))
                    vw.write(visualization)
                    key = cv2.waitKey(1)
                    index += 1
                    pbar.update(1)

                    if key == 27:
                        break
                else:
                    break

            pbar.close()
            vc.release()
            vw.release()
            cv2.destroyAllWindows()
            tracks.save(output_path, "cvat")
        except:
            print("Something went wrong...")
