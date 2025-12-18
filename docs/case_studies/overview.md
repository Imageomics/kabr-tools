# Case Studies 

## Case Study 0: Zebra Time Budgets
Example scripts for creating individual time-budgets and gantt charts from raw video data.

[Time budget notebook](0_time_budget/time_budget.ipynb)

## Case Study 1: Grevy's Landscape of Fear
Comparing effects of herd size and habitat type on the landscape of fear for Grevy's zebras.

[Statistical analysis notebook](1_grevys_landscape/grevys_landscape_lr.ipynb) 

[Visualization notebook](1_grevys_landscape/grevys_landscape_graphs.ipynb)

[Data](https://github.com/Imageomics/kabr-tools/blob/main/docs/case_studies/1_grevys_landscape/data/grevystimebudgetscleaned.csv) — Raw time-budget summaries (per bout / per individual) for Grevy's landscape case study.

This `grevystimebudgetcleaned.csv` file combines the following dates/sessions published in the [KABR Drone Wildlife Monitoring Dataset](https://huggingface.co/datasets/imageomics/kabr-behavior-telemetry) and [KABR Worked Examples](https://huggingface.co/datasets/imageomics/kabr-worked-examples/) datasets:
- KABR Drone Wildlife Monitoring Dataset: sourced from [`data/consolidated_metadata.csv`](https://huggingface.co/datasets/imageomics/kabr-behavior-telemetry/blob/main/data/consolidated_metadata.csv), specifically the sessions on `11_01_23` , `12_01_23`, and `16_01_23`.
- KABR Worked Examples: sourced from [`behavior/`](https://huggingface.co/datasets/imageomics/kabr-worked-examples/tree/main/behavior), specifically the sessions on `18_01_23`, `20_01_23`, and `21_02_23`.

For both datasets, the dates are the prefixes to the relevant videos.

## Case Study 2: Zebra State Transitions
Calculates the transition probabilities for zebra behaviors (Grevy's and Plains).

[Behavior transitions notebook](2_zebra_transition/behaviortransitionsheatmap.ipynb)

[Data on Hugging Face](https://huggingface.co/datasets/imageomics/kabr-behavior-telemetry/blob/main/data/consolidated_metadata.csv)

## Case Study 3: Mixed Species Social Interactions
Calculates the spatial overlap between different species (Grevy's zebras, plains zebras, and giraffes) and visualizes the results.

[Notebook](3_mixed_species_social/mixed_species_overlap.ipynb)

Download data used for this case-study from the [KABR worked examples dataset on Hugging Face](https://huggingface.co/datasets/imageomics/kabr-worked-examples). Use the CSV files starting with '21_01_2023_session_5' from the [`detections/`](https://huggingface.co/datasets/imageomics/kabr-worked-examples/tree/main/detections) folder.
