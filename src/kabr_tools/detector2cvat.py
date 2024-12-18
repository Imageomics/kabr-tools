import os
import argparse
import cv2
from tqdm import tqdm
from kabr_tools.utils.yolo import YOLOv8
from kabr_tools.utils.tracker import Tracker, Tracks
from kabr_tools.utils.object import Object
from kabr_tools.utils.draw import Draw


def detector2cvat(path_to_videos: str, path_to_save: str, show: bool) -> None:
    """
    Detect objects with Ultralytics YOLO detections, apply SORT tracking and convert tracks to CVAT format.

    Parameters:
    path_to_videos - str. Path to the folder containing videos.
    path_to_save - str. Path to the folder to save output xml & mp4 files.
    show - bool. Flag to display detector's visualization.
    """
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
                    if show:
                        cv2.imshow("detector2cvat", cv2.resize(
                            visualization, (int(width // 2.5), int(height // 2.5))))
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


def parse_args() -> argparse.Namespace:
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument(
        "--video",
        type=str,
        help="path to folder containing videos",
        required=True
    )
    local_parser.add_argument(
        "--save",
        type=str,
        help="path to save output xml & mp4 files",
        required=True
    )
    local_parser.add_argument(
        "--imshow",
        action="store_true",
        help="flag to display detector's visualization"
    )
    return local_parser.parse_args()


def main() -> None:
    args = parse_args()
    detector2cvat(args.video, args.save, args.imshow)


if __name__ == "__main__":
    main()
