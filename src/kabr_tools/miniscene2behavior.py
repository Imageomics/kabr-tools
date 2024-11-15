import sys
import argparse
import torch
from lxml import etree
import pandas as pd
import cv2
from tqdm import tqdm
from transformers import AutoConfig, AutoModel
from kabr_tools.utils.slowfast import get_input_clip


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

    return local_parser.parse_args()


def create_model(config_path: str, checkpoint_path: str, gpu_num: int) -> tuple[AutoConfig, torch.nn.Module]:
    # load model config
    config = AutoConfig.from_pretrained("zhong-al/x3d", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhong-al/x3d", trust_remote_code=True)
    return config, model


def annotate_miniscene(cfg: AutoConfig, model: torch.nn.Module,
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
    for track in root.iterfind("track"):
        track_id = track.attrib["id"]
        tracks.append(track_id)

    # find all frames
    # TODO: rewrite - some tracks may have different frames
    assert len(tracks) > 0, "No tracks found in track file"
    frames = []
    for box in track.iterfind("box"):
        frames.append(int(box.attrib["frame"]))

    # run model on miniscene
    for track in tracks:
        video_file = f"{miniscene_path}/{track}.mp4"
        cap = cv2.VideoCapture(video_file)
        for frame in tqdm(frames, desc=f"{track} frames"):
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

            label_data.append({"video": video,
                               "track": track,
                               "frame": frame,
                               "label": torch.argmax(preds).item()})
            if frame % 20 == 0:
                pd.DataFrame(label_data).to_csv(
                    output_path, sep=" ", index=False)
    pd.DataFrame(label_data).to_csv(output_path, sep=" ", index=False)


def main() -> None:
    # clear arguments to avoid slowfast parsing issues
    args = parse_args()
    sys.argv = [sys.argv[0]]
    cfg, model = create_model(args.config, args.checkpoint, args.gpu_num)
    annotate_miniscene(cfg, model, args.miniscene,
                       args.video, args.output)


if __name__ == "__main__":
    main()
