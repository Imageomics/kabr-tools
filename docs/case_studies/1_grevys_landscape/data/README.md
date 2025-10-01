`grevystimebudgetscleaned.csv` — Raw time-budget summaries (per bout / per individual) for Grevy's landscape case study.


The `grevystimebudgetcleaned.csv` file in `data/` combines the following dates/session published in the [KABR Datapalooza 2023 Subset](https://huggingface.co/datasets/imageomics/kabr-datapalooza-2023-subset) and [KABR Worked Examples](https://huggingface.co/datasets/imageomics/kabr-worked-examples/) datasets:
- KABR Datapalooza 2023 Subset: sourced from [`data/consolidated_metadata.csv`](https://huggingface.co/datasets/imageomics/kabr-datapalooza-2023-subset/blob/main/data/consolidated_metadata.csv), specifically the sessions on `11_01_23` , `12_01_23`, and `16_01_23`.
- KABR Worked Examples: sourced from [`behavior/`](https://huggingface.co/datasets/imageomics/kabr-worked-examples/tree/main/behavior), specifically the sessions on `18_01_23`, `20_01_23`, and `21_02_23`.

For both datasets, the dates are the prefixes to the relevant videos.
