# kabr-tools [![DOI](https://zenodo.org/badge/805519058.svg)](https://zenodo.org/doi/10.5281/zenodo.11288083)

This repository contains tools for the KABR dataset preparation.

![](https://user-images.githubusercontent.com/11778655/236357196-c09547fc-0e6b-4b2e-a7a5-18683dc944e5.png)

detector2cvat.py:\
Detect objects with Ultralytics YOLO detections, apply SORT tracking and convert tracks to CVAT format.

```
python detector2cvat.py --video path_to_videos --save path_to_save
```

cvat2ultralytics.py:\
Convert CVAT annotations to Ultralytics YOLO dataset.

```
python cvat2ultralytics.py --video path_to_videos --annotation path_to_annotations --dataset dataset_name [--skip skip_frames]
```

tracks_extractor.py:\
Extract mini-scenes from CVAT tracks.

```
python tracks_extractor.py --video path_to_videos --annotation path_to_annotations [--tracking]
```

player.py:\
Player for track and behavior observation.

```
python player.py --folder path_to_folder [--save]
```


cvat2slowfast.py:\
Convert CVAT annotations to the dataset in Charades format.

```
python cvat2slowfast.py --miniscene path_to_mini_scenes --dataset dataset_name --classes path_to_classes_json --old2new path_to_old2new_json
```
