import os
from typing import Optional
import argparse
import json
import cv2
from ruamel.yaml import YAML
from lxml import etree
from collections import OrderedDict
from tqdm import tqdm
import shutil
from natsort import natsorted


def cvat2ultralytics(video_path: str, annotation_path: str,
                     dataset: str, skip: int,
                     label2index: Optional[dict] = None) -> None:
    """
    Convert CVAT annotations to Ultralytics YOLO dataset.

    Parameters:
    video_path - str. Path to the folder containing video mp4 files.
    annotation_path - str. Path to the folder containing annotation xml files.
    dataset - str. Path to the output dataset files.
    skip - int. Process one out of skip number of frames.
    label2index - dict [optional]. Mapping of ethogram labels to integers.
    """
    # Create a YOLO dataset structure.
    dataset_file = f"""
    path: {dataset}
    train: images/train
    val: images/val
    test: images/test

    nc: 1
    names: ['Animal']
    """

    if os.path.exists(f"{dataset}"):
        shutil.rmtree(f"{dataset}")

    with open(f"{dataset}.yaml", "w") as file:
        yaml = YAML(typ='rt')
        yaml.preserve_quotes = True
        data = yaml.load(dataset_file)
        yaml.dump(data, file)

    if not os.path.exists(f"{dataset}/images/train"):
        os.makedirs(f"{dataset}/images/train")
    if not os.path.exists(f"{dataset}/images/val"):
        os.makedirs(f"{dataset}/images/val")
    if not os.path.exists(f"{dataset}/images/test"):
        os.makedirs(f"{dataset}/images/test")
    if not os.path.exists(f"{dataset}/labels/train"):
        os.makedirs(f"{dataset}/labels/train")
    if not os.path.exists(f"{dataset}/labels/val"):
        os.makedirs(f"{dataset}/labels/val")
    if not os.path.exists(f"{dataset}/labels/test"):
        os.makedirs(f"{dataset}/labels/test")

    if label2index is None:
        label2index = {
            "Grevy": 0,
            "Zebra": 0,
            "Baboon": 1,
            "Giraffe": 2
        }

    print("Process CVAT annotations...")
    videos = []
    annotations = []

    for root, dirs, files in os.walk(annotation_path):
        for file in files:
            video_name = os.path.join(video_path + root[len(annotation_path):], os.path.splitext(file)[0])
            if file.endswith(".xml"):
                if os.path.exists(video_name + ".MP4"):
                    videos.append(video_name + ".MP4")
                else:
                    videos.append(video_name + ".mp4")
                annotations.append(os.path.join(root, file))

    for i, (video, annotation) in enumerate(zip(videos, annotations)):
        print(f"{i + 1}/{len(annotations)}:", flush=True)

        if not os.path.exists(video):
            print(f"Path {video} does not exist.")
            continue

        if not os.path.exists(annotation):
            print(f"Path {annotation} does not exist.")
            continue

        # Parse CVAT for video 1.1 annotation file.
        root = etree.parse(annotation).getroot()
        name = os.path.splitext(video.split("/")[-1])[0]

        if root.find("meta").find("task") is not None:
            annotated_size = int("".join(root.find("meta").find("task").find("size").itertext()))
            width = int("".join(root.find("meta").find("task").find("original_size").find("width").itertext()))
            height = int("".join(root.find("meta").find("task").find("original_size").find("height").itertext()))
        else:
            annotated_size = int("".join(root.find("meta").find("job").find("size").itertext()))
            width = int("".join(root.find("meta").find("original_size").find("width").itertext()))
            height = int("".join(root.find("meta").find("original_size").find("height").itertext()))

        annotated = dict()
        track2end = {}

        for track in root.iterfind("track"):
            track_id = int(track.attrib["id"])
            label = label2index[track.attrib["label"].lower().capitalize()]

            for box in track.iter("box"):
                frame_id = int(box.attrib["frame"])
                keyframe = int(box.attrib["keyframe"])

                if keyframe == 1:
                    track2end[track_id] = frame_id

        for track in root.iterfind("track"):
            track_id = int(track.attrib["id"])
            label = label2index[track.attrib["label"].lower().capitalize()]

            for box in track.iter("box"):
                frame_id = int(box.attrib["frame"])

                if annotated.get(frame_id) is None:
                    annotated[frame_id] = OrderedDict()

                if frame_id <= track2end[track_id]:
                    x_start = float(box.attrib["xtl"])
                    y_start = float(box.attrib["ytl"])
                    x_end = float(box.attrib["xbr"])
                    y_end = float(box.attrib["ybr"])
                    x_center = (x_start + (x_end - x_start) / 2) / width
                    y_center = (y_start + (y_end - y_start) / 2) / height
                    w = (x_end - x_start) / width
                    h = (y_end - y_start) / height
                    annotated[frame_id][track_id] = [label, x_center, y_center, w, h]

        index = 0
        vc = cv2.VideoCapture(video)
        pbar = tqdm(total=annotated_size)

        while vc.isOpened():
            returned, frame = vc.read()
            saved = False

            if returned:
                if index > max(track2end.values()):
                    pbar.update(annotated_size - index)
                    break

                if annotated.get(index) is not None:
                    if index % skip == 0:
                        for box in annotated[index].values():
                            if not saved:
                                cv2.imwrite(f"{dataset}/images/train/{name}_{index}.jpg", frame)
                                saved = True

                            with open(f"{dataset}/labels/train/{name}_{index}.txt", "a") as file:
                                file.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

                index += 1
                pbar.update(1)
            else:
                break

        pbar.close()
        vc.release()

    print("Distribute train, val, and test...")
    images = natsorted([file for file in os.listdir(f"{dataset}/images/train") if
                        os.path.isfile(os.path.join(f"{dataset}/images/train", file))])
    labels = natsorted([file for file in os.listdir(f"{dataset}/labels/train") if
                        os.path.isfile(os.path.join(f"{dataset}/labels/train", file))])

    for file in tqdm(images[int(len(images) * 0.8):int(len(images) * 0.87)]):
        shutil.move(f"{dataset}/images/train/{file}", f"{dataset}/images/val/{file}")

    for file in tqdm(labels[int(len(labels) * 0.8):int(len(labels) * 0.87)]):
        shutil.move(f"{dataset}/labels/train/{file}", f"{dataset}/labels/val/{file}")

    for file in tqdm(images[int(len(images) * 0.87):]):
        shutil.move(f"{dataset}/images/train/{file}", f"{dataset}/images/test/{file}")

    for file in tqdm(labels[int(len(labels) * 0.87):]):
        shutil.move(f"{dataset}/labels/train/{file}", f"{dataset}/labels/test/{file}")


def parse_args() -> argparse.Namespace:
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument(
        "--video",
        type=str,
        help="path to folder containing video mp4 files",
        required=True
    )
    local_parser.add_argument(
        "--annotation",
        type=str,
        help="path to folder containing annotation xml files",
        required=True
    )
    local_parser.add_argument(
        "--dataset",
        type=str,
        help="path to output dataset files",
        required=True
    )
    local_parser.add_argument(
        "--skip",
        type=int,
        help="process one out of skip number of frames",
        default=10
    )
    local_parser.add_argument(
        "--label2index",
        type=str,
        help="path to label to index json (default is for zebra, baboon, and giraffe)",
        required=False
    )
    return local_parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.label2index:
        with open(args.label2index, mode="r", encoding="utf-8") as file:
            label2index = json.load(file)
    else:
        label2index = None

    cvat2ultralytics(args.video, args.annotation, args.dataset, args.skip, label2index)


if __name__ == "__main__":
    main()
