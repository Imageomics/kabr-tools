import os
import sys
import argparse
import json
from lxml import etree
from collections import OrderedDict
import pandas as pd
from natsort import natsorted
import cv2


def cvat2slowfast(path_to_mini_scenes, path_to_new_dataset, classes_json, old2new_json):
    with open(classes_json, mode='r', encoding='utf-8') as file:
        label2number = json.load(file)

    number2label = {value: key for key, value in label2number.items()}

    with open(old2new_json, mode='r', encoding='utf-8') as file:
        old2new = json.load(file)
        old2new[None] = None

    if not os.path.exists(path_to_new_dataset):
        os.makedirs(path_to_new_dataset)

    if not os.path.exists(f"{path_to_new_dataset}/annotation"):
        os.makedirs(f"{path_to_new_dataset}/annotation")

    if not os.path.exists(f"{path_to_new_dataset}/dataset/image"):
        os.makedirs(f"{path_to_new_dataset}/dataset/image")

    with open(f"{path_to_new_dataset}/annotation/classes.json", "w") as file:
        json.dump(label2number, file)

    headers = {"original_vido_id": [], "video_id": pd.Series(dtype="int"), "frame_id": pd.Series(dtype="int"),
               "path": [], "labels": []}
    charades_df = pd.DataFrame(data=headers)
    video_id = 1
    folder_name = 1
    flag = False

    for i, folder in enumerate(natsorted(os.listdir(path_to_mini_scenes))):
        if os.path.exists(f"{path_to_mini_scenes}/{folder}/actions"):
            for j, file in enumerate(natsorted(os.listdir(f"{path_to_mini_scenes}/{folder}/actions"))):
                if os.path.splitext(file)[1] == ".xml":
                    annotation_file = f"{path_to_mini_scenes}/{folder}/actions/{file}"
                    video_file = f"{path_to_mini_scenes}/{folder}/{os.path.splitext(file)[0]}.mp4"

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
                        if old2new[value] in label2number.keys():
                            counter += 1

                    if counter < 90:
                        print(f"SKIPPED: {folder}/actions/{file}, length={counter}<90")
                        continue

                    folder_code = f"{label[0].capitalize()}{folder_name:04d}"
                    folder_name += 1
                    output_folder = f"{path_to_new_dataset}/dataset/image/{folder_code}"
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
                            f"{path_to_new_dataset}/annotation/data.csv", sep=" ", index=False)

    charades_df.to_csv(
        f"{path_to_new_dataset}/annotation/data.csv", sep=" ", index=False)


def parse_args():
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument(
        '--miniscene',
        type=str,
        help='path to folder containing mini-scene files',
        required=True
    )
    local_parser.add_argument(
        '--dataset',
        type=str,
        help='path to output dataset files',
        required=True
    )
    local_parser.add_argument(
        '--classes',
        type=str,
        help='path to ethogram class labels json',
        required=True
    )
    local_parser.add_argument(
        '--old2new',
        type=str,
        help='path to old to new ethogram labels json',
        required=True
    )
    return local_parser.parse_args()


def main():
    args = parse_args()
    cvat2slowfast(args.miniscene, args.dataset, args.classes, args.old2new)


if __name__ == "__main__":
    main()