import os
import re
import cv2

current_dir = os.getcwd()
pattern = re.compile(r'\d+\.mp4$')

# for root, dirs, files in os.walk(current_dir):
#     video_name = root
#     print(video_name.split('|')[-1])
#     print(files)
#     mp4s = list(filter(pattern.match, files))
#     try:
#         print(mp4s[0].split('.')[0])
#     except IndexError:
#         continue
    
current_dir = os.getcwd()
pattern = re.compile(r'\d+\.mp4$')
done = False

for root, _, files in os.walk(current_dir):
    video_name = root.split('|')[-1]
    video_list = list(filter(pattern.match, files))
    print(f'l: {video_list}')
    # run model on miniscene
    for video in video_list:
        print(f"!: {video.split('.')}")
        video_file = f"{root}/{video}"
        cap = cv2.VideoCapture(video_file)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(video_file, frames)
        try:
            track = int(video.split('.')[0])
        except TypeError:
            continue