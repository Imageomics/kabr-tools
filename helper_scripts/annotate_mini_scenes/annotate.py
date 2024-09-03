# This script is used to annotate the mini_scenes using the SlowFast model. 
# It is a standalone script that can be run on a local machine. 
# It uses the SlowFast model to predict the action label for each frame in the mini_scene. 
# The script takes the path to the model config file, the model checkpoint file, the number of GPUs to use, the path to the mini_scene folder, and the output file path as arguments. 
# It then loads the model, reads the tracks.xml file in the mini_scene folder to get the list of tracks and frames, 
# and runs the model on each frame of each track in the mini_scene. 
# The predicted labels are saved to the output file as a CSV file. 
# The script also prints the progress using the tqdm library.

import sys
import torch
from lxml import etree
import pandas as pd
import cv2
import argparse
from tqdm import tqdm
import slowfast.utils.checkpoint as cu
import slowfast.models.build as build
import slowfast.utils.parser as parser
from slowfast.datasets.utils import get_sequence
from slowfast.visualization.utils import process_cv2_inputs
from slowfast.datasets.cv2_transform import scale


def get_input_clip(cap, cfg, keyframe_idx):
    # https://github.com/facebookresearch/SlowFast/blob/bac7b672f40d44166a84e8c51d1a5ba367ace816/slowfast/visualization/ava_demo_precomputed_boxes.py
    seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seq = get_sequence(
        keyframe_idx,
        seq_length // 2,
        cfg.DATA.SAMPLING_RATE,
        total_frames,
    )
    clip = []
    for frame_idx in seq:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        was_read, frame = cap.read()
        if was_read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = scale(cfg.DATA.TEST_CROP_SIZE, frame)
            clip.append(frame)
        else:
            print('Unable to read frame. Duplicating previous frame.')
            clip.append(clip[-1])

    clip = process_cv2_inputs(clip, cfg)
    return clip


def parse_args():
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument(
        '--config',
        type=str,
        help='model config.yml filepath',
        default='config.yml'
    )
    local_parser.add_argument(
        '--checkpoint',
        type=str,
        help='model checkpoint.pyth filepath',
        required=True
    )
    local_parser.add_argument(
        '--gpu_num',
        type=int,
        help='number of gpus',
        default=0
    )
    local_parser.add_argument(
        '--miniscene',
        type=str,
        help='miniscene folder containing miniscene\'s tracks.xml & *.mp4',
        required=True
    )
    local_parser.add_argument(
        '--output',
        type=str,
        help='filepath for output csv',
        default='annotation_data.csv'
    )

    return local_parser.parse_args()


def create_model(config_path, checkpoint_path, gpu_num):
    # load model config
    try:
        cfg = parser.load_config(parser.parse_args(), config_path)
    except FileNotFoundError:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'))
        with open(config_path, 'w') as file:
            file.write(checkpoint['cfg'])
        cfg = parser.load_config(parser.parse_args(), config_path)
    cfg.NUM_GPUS = gpu_num
    cfg.OUTPUT_DIR = ''
    model = build.build_model(cfg)

    # load model checkpoint
    cu.load_checkpoint(checkpoint_path, model, data_parallel=False)

    # set model to eval mode
    model.eval()
    return cfg, model


def annotate_miniscene(cfg, model, miniscene_path, video, output_path):
    label_data = []
    track_file = f'{miniscene_path}/metadata/{video}_tracks.xml'
    root = etree.parse(track_file).getroot()

    # find all tracks
    tracks = []
    for track in root.iterfind("track"):
        track_id = track.attrib["id"]
        tracks.append(track_id)

    # find all frames
    frames = []
    for box in track.iterfind("box"):
        frames.append(int(box.attrib['frame']))

    # run model on miniscene
    for track in tracks:
        video_file = f"{miniscene_path}/{track}.mp4"
        cap = cv2.VideoCapture(video_file)
        for frame in tqdm(frames, desc=f'{track} frames'):
            inputs = get_input_clip(cap, cfg, frame)

            if cfg.NUM_GPUS:
                # transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)

            preds = model(inputs)
            preds = preds.detach()

            if cfg.NUM_GPUS:
                preds = preds.cpu()

            label_data.append({'video': video,
                               'track': track,
                               'frame': frame,
                               'label': torch.argmax(preds).item()})
            if frame % 20 == 0:
                pd.DataFrame(label_data).to_csv(
                    output_path, sep=' ', index=False)
    pd.DataFrame(label_data).to_csv(output_path, sep=' ', index=False)


def main():
    # clear arguments to avoid slowfast parsing issues
    args = parse_args()
    sys.argv = [sys.argv[0]]
    print(repr(args), flush=True)
    cfg, model = create_model(args.config, args.checkpoint, args.gpu_num)
    annotate_miniscene(cfg, model, args.miniscene,
                       args.miniscene.split('/')[-1], args.output)


if __name__ == '__main__':
    main()