# Dataset Card for `kabr-tools testing`
---
## Sources of Dataset
The data is sourced from the Hugging Face dataset: [kabr_testing](https://huggingface.co/datasets/imageomics/kabr_testing/tree/main)

The raw, unprocessed drone video footage are from the following sources:
- DJI_0068 is from [18_01_23_session_7](https://huggingface.co/datasets/imageomics/KABR-raw-videos/tree/main/18_01_2023_session_7) from [KABR raw videos.](https://huggingface.co/datasets/imageomics/KABR-raw-videos)
- DJI_0001 is from [16_01_23_flight_1-DJI_0001](https://huggingface.co/datasets/imageomics/KABR-mini-scene-raw-videos/tree/main/16_01_23_flight_1-DJI_0001) from [KABR mini-scene raw videos.](https://huggingface.co/datasets/imageomics/KABR-mini-scene-raw-videos)

This dataset contains *snippets* from the orginal dataset.

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
|  ├─ DJI_0001_tracks.xml
|  └─ DJI_0001.mp4
├─ DJI_0068/
|  ├─ DJI_0068.mp4
|  └─ DJI_0068.xml
└─ README.md
```