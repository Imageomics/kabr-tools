import os
import sys
import json
from lxml import etree
from collections import OrderedDict
import pandas as pd
from natsort import natsorted
import cv2

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python cvat2slowfast.py path_to_mini_scenes")
        exit(0)
    elif len(sys.argv) == 3:
        path_to_mini_scenes = sys.argv[1]
        path_to_new_dataset = sys.argv[2]

    label2number = {"Walk": 0,
                    "Graze": 1,
                    "Browse": 2,
                    "Head Up": 3,
                    "Auto-Groom": 4,
                    "Trot": 5,
                    "Run": 6,
                    "Occluded": 7}

    number2label = {value: key for key, value in label2number.items()}

    old2new = {"Walk": "Walk",
               "Walking": "Walk",
               "Graze": "Graze",
               "Browsing": "Browse",
               "Head Up": "Head Up",
               "Auto-Groom": "Auto-Groom",
               "Mutual Grooming": "Mutual Grooming",
               "Trotting": "Trot",
               "Running": "Run",
               "Drinking": "Drinking",
               "Herding": "Herding",
               "Lying Down": "Lying Down",
               "Mounting-Mating": "Mounting-Mating",
               "Sniff": "Sniff",
               "Urinating": "Urinating",
               "Defecating": "Defecating",
               "Dusting": "Dusting",
               "Fighting": "Fighting",
               "Chasing": "Chasing",
               "Occluded": "Occluded",
               "Out of Focus": "Out of Focus",
               "Out of Frame": "Out of Frame",
               None: None}

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
                        charades_df.to_csv(f"{path_to_new_dataset}/annotation/data.csv", sep=" ", index=False)

    charades_df.to_csv(f"{path_to_new_dataset}/annotation/data.csv", sep=" ", index=False)
