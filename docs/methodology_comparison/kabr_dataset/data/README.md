# Data used in this analysis was derived from KABR mini-scene dataset

File descriptions:
### miniscene_session_summary.csv 
- Contains summary information for each mini-scene session
- Columns: `Session`, `Date`, `Start Time`, `End Time`, `Duration`, `Species`, `Herd Size`, `Video File Names`, `Weather`, `Bitterlich Score`, and `Field Notes`.

### video_miniscene_id.csv
- Contains mapping of video file names to mini-scene IDs
- Columns: Video File Name, Mini-scene ID

### video_species.csv
- Contains mapping of video file names to species (Grevy's, Plains, or Giraffes)
- Columns: Video File Name, Species


## Data source:

The KABR mini-scene dataset in HF here: https://huggingface.co/datasets/imageomics/KABR
Citation:
@misc{KABR_Data,
  author = {Kholiavchenko, Maksim and Kline, Jenna and Ramirez, Michelle and Stevens, Sam and Sheets, Alec and Babu, Reshma and Banerji, Namrata and Campolongo, Elizabeth and Thompson, Matthew and Van Tiel, Nina and Miliko, Jackson and Bessa, Eduardo and Duporge, Isla and Berger-Wolf, Tanya and Rubenstein, Daniel and Stewart, Charles},
  title = {KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos},
  year = {2023},
  url = {https://huggingface.co/datasets/imageomics/KABR},
  doi = {10.57967/hf/1010},
  publisher = {Hugging Face}
}

Consolidated metadata contains summary of mini-scenes, along with telemetry data and annotations.
https://huggingface.co/datasets/imageomics/kabr-behavior-telemetry
Citation: 
@misc{kline2024integratingbiologicaldataautonomous,
      title={Integrating Biological Data into Autonomous Remote Sensing Systems for In Situ Imageomics: A Case Study for Kenyan Animal Behavior Sensing with Unmanned Aerial Vehicles (UAVs)}, 
      author={Jenna M. Kline and Maksim Kholiavchenko and Otto Brookes and Tanya Berger-Wolf and Charles V. Stewart and Christopher Stewart},
      year={2024},
      eprint={2407.16864},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.16864}, 
}
