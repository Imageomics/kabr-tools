# kabr-tools [![DOI](https://zenodo.org/badge/805519058.svg)](https://zenodo.org/doi/10.5281/zenodo.11288083)

This repository contains tools for the KABR dataset preparation.

![](https://user-images.githubusercontent.com/11778655/236357196-c09547fc-0e6b-4b2e-a7a5-18683dc944e5.png)


These tools can be installed with:
```
pip install git+https://github.com/Imageomics/kabr-tools
```

Each KABR tool can be run through the command line (as described below) or imported as a python module. They each have help information which can be accessed on the command line through `<tool-name> -h`.

**detector2cvat:**\
Detect objects with Ultralytics YOLO detections, apply SORT tracking and convert tracks to CVAT format.

```
detector2cvat --video path_to_videos --save path_to_save
```

**cvat2ultralytics:**\
Convert CVAT annotations to Ultralytics YOLO dataset.

```
cvat2ultralytics --video path_to_videos --annotation path_to_annotations --dataset dataset_name [--skip skip_frames]
```

**tracks_extractor:**\
Extract mini-scenes from CVAT tracks.

```
tracks_extractor --video path_to_videos --annotation path_to_annotations [--tracking]
```

**player:**\
Player for track and behavior observation.

```
player --folder path_to_folder [--save]
```


**cvat2slowfast:**\
Convert CVAT annotations to the dataset in Charades format.

```
cvat2slowfast --miniscene path_to_mini_scenes --dataset dataset_name --classes path_to_classes_json --old2new path_to_old2new_json
```
