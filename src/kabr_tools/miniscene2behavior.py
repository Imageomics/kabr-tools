import sys
import argparse
import random
import torch
from lxml import etree
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from kabr_tools.utils.slowfast.utils import get_input_clip
from kabr_tools.utils.slowfast.cfg import load_config, CfgNode
from kabr_tools.utils.slowfast.x3d import build_model


def parse_args() -> argparse.Namespace:
    local_parser = argparse.ArgumentParser()
    local_parser.add_argument(
        "--config",
        type=str,
        help="model config.yml filepath",
        default="config.yml"
    )
    local_parser.add_argument(
        "--checkpoint",
        type=str,
        help="model checkpoint.pyth filepath",
        required=True
    )
    local_parser.add_argument(
        "--gpu_num",
        type=int,
        help="number of gpus",
        default=0
    )
    local_parser.add_argument(
        "--miniscene",
        type=str,
        help="miniscene folder containing miniscene\'s tracks.xml & *.mp4",
        required=True
    )
    local_parser.add_argument(
        "--video",
        type=str,
        help="name of video (expect video_tracks.xml from tracks_extractor)",
        required=True
    )
    local_parser.add_argument(
        "--output",
        type=str,
        help="filepath for output csv",
        default="annotation_data.csv"
    )
    local_parser.add_argument(
        "--slowfast",
        action="store_true",
        help="load slowfast model"
    )

    return local_parser.parse_args()


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def create_slowfast(config_path: str, checkpoint_path: str, gpu_num: int) -> tuple[CfgNode, torch.nn.Module]:
    import slowfast.utils.checkpoint as cu
    from slowfast.models import build
    from slowfast.utils import parser

    # load model config
    try:
        cfg = parser.load_config(parser.parse_args(), config_path)
    except FileNotFoundError:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"))
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(checkpoint["cfg"])
        cfg = parser.load_config(parser.parse_args(), config_path)
    cfg.NUM_GPUS = gpu_num
    cfg.OUTPUT_DIR = ""

    # set random seeds
    set_seeds(cfg.RNG_SEED)

    # load model checkpoint
    model = build.build_model(cfg)
    cu.load_checkpoint(checkpoint_path, model, data_parallel=False)

    # set model to eval mode
    model.eval()
    return cfg, model


def create_model(config_path: str, checkpoint_path: str, gpu_num: int) -> tuple[CfgNode, torch.nn.Module]:
    # load model checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True,
                            map_location=torch.device("cpu"))

    # load model config
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(checkpoint["cfg"])
        cfg = load_config(config_path)
    cfg.NUM_GPUS = gpu_num
    cfg.OUTPUT_DIR = ""

    # set random seeds
    set_seeds(cfg.RNG_SEED)

    # load model
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state"])

    # set model to eval mode
    model.eval()
    return cfg, model


def annotate_miniscene(cfg: CfgNode, model: torch.nn.Module,
                       miniscene_path: str, video: str,
                       output_path: str) -> None:
    """
    Label the mini-scenes.

    Parameters:
    cfg - CfgNode. Slowfast model configuration.
    model - torch.nn.Module. Slowfast model to use for behavior labeling.
    miniscene_path - str. Path to mini-scene folder.
    video - str. Name of video that miniscenes were extracted from.
    output_path - str. Path to save output csv.
    """
    label_data = []
    track_file = f"{miniscene_path}/metadata/{video}_tracks.xml"
    root = etree.parse(track_file).getroot()

    # find all tracks
    tracks = []
    frames = {}
    for track in root.iterfind("track"):
        track_id = track.attrib["id"]
        tracks.append(track_id)
        frames[track_id] = []

        # find all frames
        for box in track.iterfind("box"):
            frames[track_id].append(int(box.attrib["frame"]))

    # run model on miniscene
    for track in tracks:
        video_file = f"{miniscene_path}/{track}.mp4"
        cap = cv2.VideoCapture(video_file)
        index = 0
        for frame in tqdm(frames[track], desc=f'{track} frames'):
            try:
                inputs = get_input_clip(cap, cfg, index)
            except AssertionError as e:
                print(e)
                break
            index += 1

            if cfg.NUM_GPUS:
                # transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i, input_clip in enumerate(inputs):
                        inputs[i] = input_clip.cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)

            preds = model(inputs)
            preds = preds.detach()

            if cfg.NUM_GPUS:
                preds = preds.cpu()

            label_data.append({"video": video,
                               "track": track,
                               "frame": frame,
                               "label": torch.argmax(preds).item()})
            if frame % 20 == 0:
                pd.DataFrame(label_data).to_csv(
                    output_path, sep=" ", index=False)
        cap.release()
    pd.DataFrame(label_data).to_csv(output_path, sep=" ", index=False)


def main() -> None:
    # clear arguments to avoid slowfast parsing issues
    args = parse_args()
    sys.argv = [sys.argv[0]]

    # load model
    if not args.slowfast:
        cfg, model = create_model(args.config, args.checkpoint, args.gpu_num)
    else:
        cfg, model = create_slowfast(
            args.config, args.checkpoint, args.gpu_num)

    # annotate
    annotate_miniscene(cfg, model, args.miniscene,
                       args.video, args.output)


if __name__ == "__main__":
    main()
