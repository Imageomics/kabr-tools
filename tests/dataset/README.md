# Test Dataset

Video snippets and associated behavior annotations were selected from the [KABR collection](https://huggingface.co/collections/imageomics/kabr) for testing elements of this package.

## Source Data

Specifically, these *snippets* were selected from the original, raw, unprocessed KABR drone video footage. The videos were specially selected and truncated for use in testing the tools in this repository, used to make the [KABR mini-scene dataset](https://huggingface.co/datasets/imageomics/KABR) with behavior annotations.

Data Sources by folder:
- `DJI_0001/` were selected from [16_01_23_flight_1-DJI_0001](https://huggingface.co/datasets/imageomics/KABR-mini-scene-raw-videos/tree/main/16_01_23_flight_1-DJI_0001) in the [KABR mini-scene raw videos dataset](https://huggingface.co/datasets/imageomics/KABR-mini-scene-raw-videos).
- `DJI_0068/` files are from [18_01_23_session_7](https://huggingface.co/datasets/imageomics/KABR-raw-videos/tree/main/18_01_2023_session_7), part of the [KABR raw videos dataset](https://huggingface.co/datasets/imageomics/KABR-raw-videos). These videos were collected during the same time period as the others, but were not selected for use in the KABR mini-scene dataset.

The file structure follows that of the two KABR raw video datasets from which these snippets were sourced.

## File Structure
```tree
dataset/
├─ DJI_0001/
|  ├─ actions/
|  |  └─ 43.xml
|  ├─ metadata/
|  |  ├─ DJI_0001_metadata.json
|  |  └─ DJI_0001_tracks.xml
|  ├─ 43.mp4
|  └─ DJI_0001.mp4
├─ DJI_0068/
|  ├─ DJI_0068.mp4
|  └─ DJI_0068.xml
└─ README.md
```