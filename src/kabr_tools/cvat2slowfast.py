import os
import sys
from typing import Optional
import argparse
import json
from lxml import etree
from collections import OrderedDict
import pandas as pd
from natsort import natsorted
import cv2
from utils.path import join_paths


def cvat2slowfast(path_to_mini_scenes: str, path_to_new_dataset: str,
                  label2number: dict, old2new: Optional[dict]) -> None:
    """
    Convert CVAT annotations to the dataset in Charades format.

    Parameters:
    path_to_mini_scenes - str. Path to the folder containing mini-scene files.
    path_to_new_dataset - str. Path to the folder to output dataset files.
    label2number - dict. Mapping of ethogram labels to integers.
    old2new - dict [optional]. Mapping of old ethogram labels to new ethogram labels.
    """
    if not os.path.exists(path_to_new_dataset):
        os.makedirs(path_to_new_dataset)

    annotation_path = join_paths(path_to_new_dataset, "annotation")
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)

    image_path = join_paths(path_to_new_dataset, "dataset", "image")
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    classes_path = join_paths(annotation_path, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as file:
        json.dump(label2number, file)

    headers = {"original_vido_id": [], "video_id": pd.Series(dtype="int"), "frame_id": pd.Series(dtype="int"),
               "path": [], "labels": []}
    charades_df = pd.DataFrame(data=headers)
    video_id = 1
    folder_name = 1
    flag = False

    for i, folder in enumerate(natsorted(os.listdir(path_to_mini_scenes))):
        actions_path = join_paths(path_to_mini_scenes, folder, "actions")
        if os.path.exists(actions_path):
            for j, file in enumerate(natsorted(os.listdir(actions_path))):
                if os.path.splitext(file)[1] == ".xml":
                    annotation_file = join_paths(actions_path, file)
                    video_file = join_paths(path_to_mini_scenes, folder, f"{os.path.splitext(file)[0]}.mp4")

                    if not os.path.exists(video_file):
                        print(f"{video_file} does not exist.")
                        continue

                    root = etree.parse(annotation_file).getroot()

                    try:
                        label = next(root.iterfind("track")).attrib["label"]
                    except StopIteration:
                        print(f"SKIPPED: {folder}/actions/{file}, EMPTY ANNOTATION")
                        continue

                    annotated = OrderedDict()

                    for track in root.iterfind("track"):
                        for entry in track.iter("points"):
                            frame_id = entry.attrib["frame"]
                            outside = entry.attrib["outside"]

                            if outside == "1":
                                continue

                            behavior = "".join(entry.find("attribute").itertext())

                            if annotated.get(frame_id) is None:
                                annotated[frame_id] = OrderedDict()

                            annotated[frame_id] = behavior

                    counter = 0

                    for value in annotated.values():
                        if old2new:
                            if old2new[value] in label2number:
                                counter += 1
                        elif (value in label2number):
                            counter += 1

                    if counter < 90:
                        print(f"SKIPPED: {folder}/actions/{file}, length={counter}<90")
                        continue

                    folder_code = f"{label[0].capitalize()}{folder_name:04d}"
                    folder_name += 1
                    output_folder = join_paths(image_path, folder_code)
                    progress = f"{i + 1}/{len(os.listdir(path_to_mini_scenes))}," \
                               f"{j + 1}/{len(os.listdir(f'{path_to_mini_scenes}/{folder}/actions'))}:" \
                               f"{folder}/actions/{file} -> {output_folder}"
                    print(progress)
                    sys.stdout.flush()

                    index = 0
                    adjusted_index = 1
                    vc = cv2.VideoCapture(video_file)
                    size = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

                    while vc.isOpened():
                        if flag is False:
                            if index < size:
                                returned = True
                                frame = None
                            else:
                                returned = False
                                frame = None
                        else:
                            returned, frame = vc.read()

                        if returned:
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

                            behavior = annotated.get(str(index))

                            if old2new:
                                behavior = old2new[behavior]

                            if behavior in label2number.keys():
                                if flag:
                                    cv2.imwrite(f"{output_folder}/{adjusted_index}.jpg", frame)

                                # TODO: Major slow down here. Add to a list rather than dataframe,
                                #  and create dataframe at the end.
                                charades_df.loc[len(charades_df.index)] = [f"{folder_code}",
                                                                           video_id,
                                                                           adjusted_index,
                                                                           f"{folder_code}/{adjusted_index}.jpg",
                                                                           str(label2number[behavior])]

                                adjusted_index += 1

                            index += 1
                        else:
                            break

                    vc.release()
                    video_id += 1

                    if video_id % 10 == 0:
                        charades_df.to_csv(
                            join_paths(annotation_path, "data.csv"), sep=" ", index=False)

    charades_df.to_csv(
        join_paths(annotation_path, "data.csv"), sep=" ", index=False)


def parse_args() -> argparse.Namespace:
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument(
        "--miniscene",
        type=str,
        help="path to folder containing mini-scene files",
        required=True
    )
    local_parser.add_argument(
        "--dataset",
        type=str,
        help="path to output dataset files",
        required=True
    )
    local_parser.add_argument(
        "--classes",
        type=str,
        help="path to ethogram class labels json",
        required=True
    )
    local_parser.add_argument(
        "--old2new",
        type=str,
        help="path to old to new ethogram labels json",
        required=False
    )
    return local_parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.classes, mode="r", encoding="utf-8") as file:
        label2number = json.load(file)

    if args.old2new:
        with open(args.old2new, mode="r", encoding="utf-8") as file:
            old2new = json.load(file)
            old2new[None] = None
    else:
        old2new = None

    cvat2slowfast(args.miniscene, args.dataset, label2number, old2new)


if __name__ == "__main__":
    main()
