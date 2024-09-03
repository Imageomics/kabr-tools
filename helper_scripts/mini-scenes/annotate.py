import os
import re
import torch
from lxml import etree
import pandas as pd
import cv2
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


def annotate_miniscene(cfg, model, output_path):
    label_data = []
    current_dir = os.getcwd()
    pattern = re.compile(r'\d+\.mp4$')
    done = False

    for root, _, files in os.walk(current_dir):
        print('c')
        print(files)
        video_name = root.split('|')[-1]
        video_list = list(filter(pattern.match, files))
        print(f'l: {video_list}')
        # run model on miniscene
        for video in video_list:
            try:
                track = int(video.split('.')[0])
            except Exception:
                continue
            
            print(root, video)
            
            video_file = f"{root}/{video}"
            cap = cv2.VideoCapture(video_file)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            print(video_file, frames)
            
            for frame in tqdm(range(frames), desc=f'{track} frames'):
                inputs = get_input_clip(cap, cfg, frame)
                print('d')

                if cfg.NUM_GPUS:
                    # transfer the data to the current GPU device.
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)

                preds = model(inputs)
                print('e')
                preds = preds.detach()
                
                print('f')

                if cfg.NUM_GPUS:
                    preds = preds.cpu()

                label_data.append({'video': video_name,
                                'track': track,
                                'frame': frame,
                                'label': torch.argmax(preds).item()})
                if frame % 20 == 0:
                    pd.DataFrame(label_data).to_csv(
                        output_path, sep=' ', index=False)
                print(label_data)
            done = True
            break
        if done:
            break
    pd.DataFrame(label_data).to_csv(output_path, sep=' ', index=False)


def main():
    cfg, model = create_model('config.yml', 'model/checkpoint_epoch_00075.pyth', 1)
    print('a')
    annotate_miniscene(cfg, model, 'annotation_data.csv')
    print('b')


if __name__ == '__main__':
    main()