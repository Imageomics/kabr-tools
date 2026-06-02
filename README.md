# kabr-tools  [![DOI]()]()
<!-- need new DOI for this release -->

#### This repository contains tools to analyze *in situ* animal behavior from video.

The modular pipeline processes video through object detection, individual tracking, and machine learning-based behavioral classification to generate ecological metrics including time budgets, behavioral transitions, land use and habitat, social interactions, and demographic data. Framework design enables integration of novel ML models and adaptation across species and study systems.

<details>
  <summary>Click to expand detailed description</summary>

Understanding community-level ecological patterns requires scalable methods to process multi-modal and multi-scalar behavioral data. Traditional field observations are limited in scope, making it difficult to assess behavioral responses across landscapes. To address this, we present Kenyan Animal Behavior Recognition, kabr-tools. This open-source computational ecology framework integrates camera-trap and drone video with machine learning to automatically extract behavioral, social, and spatial metrics from wildlife footage.

Our pipeline processes multi-species camera trap and drone data using object detection, tracking, and behavioral classification to generate five key metrics: time budgets, behavioral transitions, social interactions, habitat associations, and group composition dynamics. The modular design allows for integration of novel machine learning models and adaptation across species and study systems. We demonstrate the utility of kabr-tools through case studies on zebra time budgets, mixed-species social interactions, and African wild dog individual identification. By automating behavioral monitoring, kabr-tools enables scalable ecological research and conservation applications across diverse ecosystems.
</details>


## Overview

![](https://imageomics.github.io/kabr-tools/images/kabr-tools-visual-abstract.png)
**Figure 1:** kabr-tools computational framework for automated wildlife behavioral monitoring. 

### 🔧 Installation

`kabr-tools` requires **Python 3.10 or 3.11**.

### Basic installation

```bash
pip install -e .
```

### Optional installation modes

- Behavior classification with SlowFast:
  ```bash
  pip install -e ".[slowfast]"
  ```

- Development tools:
  ```bash
  pip install -e ".[dev]"
  ```

- Documentation tools:
  ```bash
  pip install -e ".[docs]"
  ```

- All-in-one (recommended for contributors):
  ```bash
  pip install -e ".[slowfast,dev,docs]"
  ```

> Note: For CUDA-specific `torch` versions, refer to [pytorch.org](https://pytorch.org/get-started/locally/).

## 📖 Project Overview

The KABR tools pipeline consists of:

1. **Video collection using drones and/or camera traps**
2. **Annotation and tracking using CVAT**
3. **Mini-scene extraction**
4. **Behavior classification**
5. **Ecological analysis**

Each script can be run via command-line (`<tool-name> -h`) or imported as a module.

## Documentation

- [User Guide](https://imageomics.github.io/kabr-tools/pipeline/overview)
- [Worked Example](https://imageomics.github.io/kabr-tools/pipeline/worked-example/)
- [CVAT User Guide](https://imageomics.github.io/kabr-tools/cvat/cvat-guide/)
- [Data Management Tips](https://imageomics.github.io/kabr-tools/cvat/cvat-data-management/)

## Methodology Comparison and Case Studies

####  See [Methodology Comparison](https://imageomics.github.io/kabr-tools/methodology_comparison/overview/) for a detailed comparison of different methods for animal behavior analysis, including:
- Focal sampling
- Scan sampling
- Drone-based video analysis

#### See [Case Studies](https://imageomics.github.io/kabr-tools/case_studies/overview/) for examples demonstrating the application of kabr-tools in various ecological contexts, including:
- Grevy's zebra time budgets
- Mixed-species social interactions
- African wild dog individual behavioral monitoring

#### Example time budget analysis output comparing behavior granularity collected from drone videos versus traditional field observations.

![graph displaying the time budget comparison between drone-based behavior classification and manual field focal observations, demonstrates how drones allow for observations of more behaviors](https://imageomics.github.io/kabr-tools/images/timebuget_drone_manual.png)
**Figure 2:** Time budget comparison between drone-based behavior classification (bottom) and manual field focal observations (top). 

## Related Papers
Please refer to our papers for details on the data collection process and machine learning model development.

- [KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition](https://openaccess.thecvf.com/content/WACV2024W/CV4Smalls/papers/Kholiavchenko_KABR_In-Situ_Dataset_for_Kenyan_Animal_Behavior_Recognition_From_Drone_WACVW_2024_paper.pdf) 
- [Deep dive into KABR: a dataset for understanding ungulate behavior from in-situ drone video](https://link.springer.com/article/10.1007/s11042-024-20512-4) 
- [Integrating Biological Data into Autonomous Remote Sensing Systems for In Situ Imageomics: A Case Study for Kenyan Animal Behavior Sensing with Unmanned Aerial Vehicles (UAVs)](https://arxiv.org/abs/2407.16864)
- [A Framework for Autonomic Computing for In Situ Imageomics](https://ieeexplore.ieee.org/abstract/document/10336017)

## Citation

If you use this toolkit in your research, please cite both this package and the associated paper ([kabr-tools: Automated Framework for Multi-Species Behavioral Monitoring](https://arxiv.org/abs/2510.02030)):

**Package citation:**
```bibtex
@software{Kline_KABR_Tools_2025,
      author = {Kline, Jenna and Zhong, Alison and Campolongo, Elizabeth and Kholiavchenko, Maksim and Thompson, Matthew J.},
      doi = {10.5281/zenodo.11288083},
      license = {MIT},
      month = dec,
      title = {{KABR Tools}},
      url = {https://github.com/Imageomics/kabr-tools},
      version = {3.0.0},
      year = {2025}
}
```

**Paper citation:**
```bibtex
@misc{kline2025kabrtoolsautomatedframeworkmultispecies,
      title={kabr-tools: Automated Framework for Multi-Species Behavioral Monitoring}, 
      author={Jenna Kline and Maksim Kholiavchenko and Samuel Stevens and Nina van Tiel and Alison Zhong and Namrata Banerji and Alec Sheets and Sowbaranika Balasubramaniam and Isla Duporge and Matthew Thompson and Elizabeth Campolongo and Jackson Miliko and Neil Rosser and Tanya Berger-Wolf and Charles V. Stewart and Daniel I. Rubenstein},
      year={2025},
      eprint={2510.02030},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.02030}, 
}
```

## 💬 Feedback & Issues

Open issues on [GitHub](https://github.com/Imageomics/kabr-tools/issues).

## 🔗 License

This project is licensed under the [MIT License](https://github.com/Imageomics/kabr-tools/blob/main/LICENSE).

![](https://imageomics.github.io/kabr-tools/images/zebras_boxes.webp)
