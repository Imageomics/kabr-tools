# Case Studies 

<!-- Description for 0_time_budget/time_budget.ipynb? -->

## Case Study 1: Grevy's Landscape of Fear
[Notebook](1_grevys_landscape/grevys_landscape_of_fear.ipynb) <!-- broken URL, no record of this notebook -->

[Data](https://github.com/Imageomics/kabr-tools/blob/main/docs/case_studies/1_grevys_landscape/data/grevystimebudgetscleaned.csv) — Raw time-budget summaries (per bout / per individual) for Grevy's landscape case study.

This `grevystimebudgetcleaned.csv` file combines the following dates/sessions published in the [KABR Drone Wildlife Monitoring Dataset](https://huggingface.co/datasets/imageomics/kabr-behavior-telemetry) and [KABR Worked Examples](https://huggingface.co/datasets/imageomics/kabr-worked-examples/) datasets:
- KABR Drone Wildlife Monitoring Dataset: sourced from [`data/consolidated_metadata.csv`](https://huggingface.co/datasets/imageomics/kabr-behavior-telemetry/blob/main/data/consolidated_metadata.csv), specifically the sessions on `11_01_23` , `12_01_23`, and `16_01_23`.
- KABR Worked Examples: sourced from [`behavior/`](https://huggingface.co/datasets/imageomics/kabr-worked-examples/tree/main/behavior), specifically the sessions on `18_01_23`, `20_01_23`, and `21_02_23`.

For both datasets, the dates are the prefixes to the relevant videos.

## Case Study 2: Zebra State Transitions
[Notebook](2_zebra_transition/behaviortransitionsheatmap.ipynb)

[Data on Hugging Face](https://huggingface.co/datasets/imageomics/kabr-behavior-telemetry/tree/main/data/consolidated_metadata.csv)

## Case Study 3: Mixed Species Social Interactions
[Notebook](3_mixed_species_social/mixed_species_overlap.ipynb)

Download data used for this case-study from the [KABR worked examples dataset on Hugging Face](https://huggingface.co/datasets/imageomics/kabr-worked-examples). Use the CSV files starting with '21_01_2023_session_5' from the [`detections/`](https://huggingface.co/datasets/imageomics/kabr-worked-examples/tree/main/detections) folder.
