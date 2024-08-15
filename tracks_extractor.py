import numpy as np
import os
import sys
import json
from lxml import etree
import shutil
import cv2
from src.utils import get_scene
from collections import OrderedDict
from src.detector import Detector
from src.tracker import Tracker, Tracks
from src.object import Object
from src.draw import Draw
from tqdm import tqdm


def generate_timeline_image(name, folder, timeline, annotated_size):
    timeline_image = np.zeros(shape=(len(timeline["tracks"].keys()) * 100, annotated_size, 3), dtype=np.uint8)

    for i, (key, value) in enumerate(timeline["tracks"].items()):
        if timeline["colors"].get(key) is None:
            color = (127, 127, 127)
        else:
            color = timeline["colors"][key]

        binary = np.array(value, dtype=np.int32)
        binary[binary >= 0] = 1
        binary[binary < 0] = 0
        timeline_image[(i * 100):(i + 1) * 100, 0:annotated_size] = color
        mask = np.repeat(np.array(binary, dtype=np.uint8).reshape(1, -1), repeats=100, axis=0)
        image = timeline_image[(i * 100):(i + 1) * 100, 0:annotated_size]
        timeline_image[(i * 100):(i + 1) * 100, 0:annotated_size] = \
            cv2.bitwise_and(image, image, mask=mask)

    timeline_resized = cv2.resize(timeline_image, (1000, timeline_image.shape[0]))

    for i, (key, value) in enumerate(timeline["tracks"].items()):
        if timeline["colors"].get(key) is None:
            color = (127, 127, 127)
        else:
            color = timeline["colors"][key]

        cv2.putText(timeline_resized, str(key), (30, i * 100 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple([j - 30 for j in color]), 2, cv2.LINE_AA)

    cv2.imwrite(f"mini-scenes/{folder}/metadata/{name}.jpg", timeline_resized)


def extract(video_path, annotation_path, tracking):
    # Parse CVAT for video 1.1 annotation file.
    root = etree.parse(annotation_path).getroot()
    annotated = dict()

    for track in root.iterfind("track"):
        track_id = int(track.attrib["id"])

        for box in track.iter("box"):
            frame_id = int(box.attrib["frame"])

            if annotated.get(frame_id) is None:
                annotated[frame_id] = OrderedDict()

            annotated[frame_id][track_id] = [int(float(box.attrib["xtl"])),
                                             int(float(box.attrib["ytl"])),
                                             int(float(box.attrib["xbr"])),
                                             int(float(box.attrib["ybr"]))]

    name = os.path.splitext(video_path.split("/")[-1])[0]
    folder = os.path.splitext("|".join(video_path.split("/")[-3:]))[0]
    annotated_size = int("".join(root.find("meta").find("task").find("size").itertext()))
    scene_width, scene_height = 400, 300
    vc = cv2.VideoCapture(video_path)
    original_width, original_height = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{video_path} | {annotation_path} -> mini-scenes/{folder}")

    if not os.path.exists(f"mini-scenes/{folder}"):
        os.makedirs(f"mini-scenes/{folder}")

    vw = cv2.VideoWriter(f"mini-scenes/{folder}/{name}.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"), 29.97,
                         (original_width, original_height))
    max_disappeared = 40
    tracker = Tracker(max_disappeared=max_disappeared, max_distance=300)
    tracks = Tracks(max_disappeared=max_disappeared, interpolation=True)
    index = 0
    tracked_indices = OrderedDict()
    timeline = OrderedDict()
    timeline["original"] = video_path
    timeline["tracks"] = OrderedDict()
    timeline["tracks"]["main"] = [-1] * annotated_size
    timeline["colors"] = {}
    vc.set(cv2.CAP_PROP_POS_FRAMES, index)
    tracks_vw = dict()
    pbar = tqdm(total=annotated_size)

    while vc.isOpened():
        returned, frame = vc.read()

        if returned:
            visualization = frame.copy()

            if annotated.get(index) is not None:
                centroids = []
                attributes = []
                objects = OrderedDict()
                colors = OrderedDict()

                for object_id, box in annotated[index].items():
                    attribute = {}
                    centroid = Detector.get_centroid(box)
                    centroids.append(centroid)
                    attribute["box"] = box
                    attributes.append(attribute)

                    if not tracking:
                        objects[object_id] = centroid
                        colors_values = list(tracker.colors_table.values())
                        colors[object_id] = colors_values[object_id % len(colors_values)]
                        timeline["colors"][object_id] = colors[object_id]

                if tracking:
                    objects, colors = tracker.update(centroids)

                objects = Object.object_factory(objects, centroids, colors, attributes=attributes)
                tracks.update(objects, index)

                for object in objects:
                    if tracks_vw.get(object.object_id) is None:
                        if not os.path.exists(f"mini-scenes/{folder}"):
                            os.makedirs(f"mini-scenes/{folder}")

                        tracks_vw[object.object_id] = cv2.VideoWriter(f"mini-scenes/{folder}/{object.object_id}.mp4",
                                                                      cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                                                      29.97, (scene_width, scene_height))
                        tracked_indices[object.object_id] = 0
                        timeline["tracks"][object.object_id] = [-1] * annotated_size

                for object in objects:
                    Draw.track(visualization, tracks[object.object_id].centroids, object.color, 20)
                    Draw.scene(visualization, object, scene_width, scene_height)
                    Draw.object_id(visualization, object)
                    scene_frame = frame.copy()
                    scene_frame = get_scene(scene_frame, object, scene_width, scene_height)
                    tracks_vw[object.object_id].write(scene_frame)
                    timeline["tracks"][object.object_id][index] = tracked_indices[object.object_id]
                    tracked_indices[object.object_id] += 1

            cv2.imshow("tracks_extractor", cv2.resize(visualization,
                                                      (int(original_width // 2.5), int(original_height // 2.5))))
            vw.write(visualization)
            key = cv2.waitKey(1)
            timeline["tracks"]["main"][index] = index
            index += 1
            pbar.update(1)

            if key == 27:
                break
        else:
            break

    for track_key in tracks_vw.keys():
        tracks_vw[track_key].release()

    if not os.path.exists(f"mini-scenes/{folder}/actions"):
        os.makedirs(f"mini-scenes/{folder}/actions")

    if not os.path.exists(f"mini-scenes/{folder}/metadata"):
        os.makedirs(f"mini-scenes/{folder}/metadata")

    shutil.copy(annotation_path, f"mini-scenes/{folder}/metadata/{name}_tracks.xml")
    generate_timeline_image(name, folder, timeline, annotated_size)

    with open(f"mini-scenes/{folder}/metadata/{name}_metadata.json", "w") as file:
        json.dump(timeline, file)

    pbar.close()
    vc.release()
    vw.release()
    cv2.destroyAllWindows()

def tracks_extractor(video, annotation, tracking):
    if os.path.isdir(annotation):
        videos = []
        annotations = []

        for root, dirs, files in os.walk(annotation):
            for file in files:
                if os.path.splitext(file)[1] == ".xml":
                    folder = root.split("/")[-1]

                    if folder.startswith("!") or file.startswith("!"):
                        continue

                    videos.append(os.path.join(video + root[len(annotation):], os.path.splitext(file)[0] + ".mp4"))
                    annotations.append(os.path.join(root, file))

        for i, (video, annotation) in enumerate(zip(videos, annotations)):
            print(f"{i + 1}/{len(annotations)}:")

            if not os.path.exists(video):
                print(f"Path {video} does not exist.")
                continue

            extract(video, annotation, tracking)
    else:
        extract(video, annotation, tracking)

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("python tracks_extractor.py path_to_videos path_to_annotations [tracking]")
        exit(0)
    elif len(sys.argv) == 3:
        video = sys.argv[1]
        annotation = sys.argv[2]
        tracking = False
    # tracking=True: use external tracker instead of CVAT tracks.
    # tracking=False: use CVAT tracks.
    elif len(sys.argv) == 4:
        video = sys.argv[1]
        annotation = sys.argv[2]
        tracking = bool(sys.argv[3])

    tracks_extractor(video, annotation, tracking)
