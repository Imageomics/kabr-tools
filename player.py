import os
import sys
import json
from lxml import etree
from collections import OrderedDict
import cv2


def on_slider_change(value):
    global index, vcs, current, trackbar_position, paused, updated
    index = value

    if abs(trackbar_position - index) > 10:
        vcs[current].set(cv2.CAP_PROP_POS_FRAMES, metadata["tracks"][current][index])

        if paused:
            updated = True


def pad(image, width, height):
    shape_0, shape_1 = image.shape[0], image.shape[1]

    if shape_0 < shape_1:
        new_width = int((height / shape_0) * shape_1)
        pad_size = (width - new_width) // 2
        image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(image, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        new_height = int((width / shape_1) * shape_0)
        pad_size = (height - new_height) // 2
        image = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded


def draw_aim(current, image):
    if current == "main":
        return image

    copied = image.copy()
    x = image.shape[1] // 2
    y = image.shape[0] // 2
    cv2.line(image, (x - 20, y), (x + 20, y), (127, 127, 127), 2)
    cv2.line(image, (x, y - 20), (x, y + 20), (127, 127, 127), 2)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


def draw_id(current, image, metadata, width):
    if current == "main":
        label = f"Drone View"
        color = (127, 127, 127)
    else:
        label = f"Track #{current}"
        color = metadata["colors"][current]

    thickness_in = 4
    size = 1.5
    label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)

    copied = image.copy()
    cv2.rectangle(image, (width // 2 - label_length[0][0] // 2 - 50, 95),
                  (width // 2 + label_length[0][0] // 2 + 50, 180), (255, 255, 255), -1)
    cv2.putText(image, label, ((width - label_length[0][0]) // 2, 150),
                cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i - 50 for i in color]), thickness_in, cv2.LINE_AA)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


def draw_actions(current, index, image, actions, metadata, width, height):
    if current == "main":
        return image

    if actions.get(current) is None:
        return image
    elif actions[current].get(str(metadata["tracks"][current][index])) is None:
        return image

    color = metadata["colors"][current]
    label = "|".join(actions[current][str(metadata["tracks"][current][index])].split(","))
    thickness_in = 4
    size = 1.5
    label_length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, thickness_in)

    copied = image.copy()
    cv2.rectangle(image, (width // 2 - label_length[0][0] // 2 - 50, height - 95),
                  (width // 2 + label_length[0][0] // 2 + 50, height - 180), (255, 255, 255), -1)
    cv2.putText(image, label, ((width - label_length[0][0]) // 2, height - 130),
                cv2.FONT_HERSHEY_SIMPLEX, size, tuple([i - 50 for i in color]), thickness_in, cv2.LINE_AA)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


def draw_info(image, width):
    copied = image.copy()
    cv2.rectangle(image, (width - 600, 100), (width - 100, 340), (0, 0, 0), -1)
    cv2.putText(image, "[0-9]: Show Track #[0-9]", (width - 565, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Shift + [0-9]: Show Track #[10-19]", (width - 565, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Enter: Show Drone View", (width - 565, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Esc: Exit", (width - 565, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(image, 0.4, copied, 0.6, 0.0)


def hotkey(key):
    global current, metadata, vc, letter2hotkey

    mapped = letter2hotkey[key]

    if mapped == "main":
        current = mapped
        vc = vcs[current]
        vc.set(cv2.CAP_PROP_POS_FRAMES, metadata["tracks"][current][index])
    elif mapped in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]:
        if metadata["tracks"].get(mapped) is not None:
            if metadata["tracks"][mapped][index] != -1:
                current = mapped
                vc = vcs[current]

                if index < len(metadata["tracks"][mapped]):
                    if metadata["tracks"][mapped][index] < 0:
                        current = "main"
                        vc = vcs[current]

                    vc.set(cv2.CAP_PROP_POS_FRAMES, metadata["tracks"][current][index])


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("python player.py path_to_folder [save]")
        exit(0)
    elif len(sys.argv) == 2:
        folder = sys.argv[1]
        save = True#False
    elif len(sys.argv) == 3:
        folder = sys.argv[1]
        save = bool(sys.argv[2])

    name = folder.split("/")[-1].split('|')[-1]

    metadata_path = f"{folder}/metadata/{name}_metadata.json"
    actions_path = f"{folder}/actions"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    vcs = OrderedDict()
    vcs["main"] = cv2.VideoCapture(f"{folder}/{name}.mp4")

    for file in os.listdir(folder):
        if os.path.splitext(file)[-1] == ".mp4":
            if not os.path.splitext(file)[0].startswith(name):
                vcs[os.path.splitext(file)[0]] = cv2.VideoCapture(f"{folder}/{file}")

    actions = OrderedDict()

    if os.path.exists(actions_path):
        for file in os.listdir(actions_path):
            if os.path.splitext(file)[-1] == ".xml":
                root = etree.parse(f"{actions_path}/{file}").getroot()
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

                    actions[os.path.splitext(file)[0]] = annotated

    index = 0
    cv2.namedWindow("TrackPlayer")
    cv2.createTrackbar(name, "TrackPlayer", index, len(metadata["tracks"]["main"]) - 1, on_slider_change)
    current = "main"
    vc = vcs[current]
    target_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    letter2hotkey = {13: "main", 48: "0", 49: "1", 50: "2", 51: "3", 52: "4", 53: "5", 54: "6", 55: "7", 56: "8",
                     57: "9", 41: "10", 33: "11", 64: "12", 35: "13", 36: "14", 37: "15", 94: "16", 38: "17",
                     42: "18", 40: "19"}
    paused, updated = False, False

    if save:
        vw = cv2.VideoWriter(f"{folder}/{name}_demo.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                             29.97, (target_width, target_height))

    while vc.isOpened():
        if index < len(metadata["tracks"][current]):
            if metadata["tracks"][current][index] < 0:
                current = "main"
                vc = vcs[current]
                vc.set(cv2.CAP_PROP_POS_FRAMES, metadata["tracks"][current][index])

        returned, frame = vc.read()

        if returned:
            updated = False
            visualization = frame.copy()

            if current != "main":
                visualization = pad(visualization, target_width, target_height)

            visualization = draw_aim(current, visualization)
            visualization = draw_id(current, visualization, metadata, target_width)
            visualization = draw_actions(current, index, visualization, actions, metadata, target_width, target_height)
            visualization = draw_info(visualization, target_width)
            trackbar_position = cv2.getTrackbarPos(name, "TrackPlayer")
            cv2.setTrackbarPos(name, "TrackPlayer", index)
            cv2.putText(visualization, f"Frame: {index}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.imshow("TrackPlayer", cv2.resize(visualization, (int(target_width // 2.5), int(target_height // 2.5)),
                                                 interpolation=cv2.INTER_AREA))

            if save:
                vw.write(visualization)

            key = cv2.waitKey(1)

            if key == 27:
                break
            elif key == 32 or paused:
                paused = True
                flag = False

                while True:
                    key = cv2.waitKey(1)

                    if key == 32:
                        paused = False
                        break
                    elif key == 27:
                        flag = True
                        break
                    elif letter2hotkey.get(key) is not None:
                        if letter2hotkey[key] in vcs.keys():
                            if metadata["tracks"][letter2hotkey[key]][index] >= 0:
                                hotkey(key)
                                break
                    elif updated:
                        break

                if flag:
                    break
            elif letter2hotkey.get(key) is not None:
                if letter2hotkey[key] in vcs.keys():
                    if metadata["tracks"][letter2hotkey[key]][index] >= 0:
                        hotkey(key)

            index += 1
        else:
            if current == "main":
                break
            else:
                current = "main"
                vc = vcs[current]
                vc.set(cv2.CAP_PROP_POS_FRAMES, metadata["tracks"][current][index])

    if save:
        vw.release()

    for k, v in vcs.items():
        v.release()

    cv2.destroyAllWindows()
