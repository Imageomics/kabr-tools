# KABR Tools Documentation

[![DOI](https://zenodo.org/badge/805519058.svg)](https://zenodo.org/doi/10.5281/zenodo.11288083)

## Overview

This repository contains tools to perform animal behavioral analysis from drone videos.

The modular pipeline processes drone video through object detection, individual tracking, and machine learning-based behavioral classification to generate ecological metrics including time budgets, behavioral transitions, land use and habitat, social interactions, and demographic data. Framework design enables integration of novel ML models and adaptation across species and study systems.

![Visual Abstract](images/visual_abstract.png)

**Figure 1:** kabr-tools computational framework for automated wildlife behavioral monitoring.

## Detailed Description

Understanding community-level ecological patterns requires scalable methods to process multi-dimensional behavioral data. Traditional field observations are limited in scope, making it difficult to assess behavioral responses across landscapes. To address this, we present Kenyan Animal Behavior Recognition, kabr-tools. This open-source computational ecology framework integrates drone-based video with machine learning to automatically extract behavioral, social, and spatial metrics from wildlife footage.

Our pipeline processes multi-species drone data using object detection, tracking, and behavioral classification to generate five key metrics: time budgets, behavioral transitions, social interactions, habitat associations, and group composition dynamics. Validated on three African species, our system achieved 65 - 70% behavioral classification accuracy, with >95% accuracy for certain behaviors.

## Installation

KABR tools requires that torch be installed.

The KABR tools used in this process can be installed with:

```bash
pip install torch torchvision
pip install git+https://github.com/Imageomics/kabr-tools
```

!!! note "PyTorch Installation"
    Refer to [pytorch.org](https://pytorch.org/get-started/locally/) to install specific versions of torch/CUDA

Each KABR tool can be run through the command line (as described below) or imported as a python module. They each have help information which can be accessed on the command line through `<tool-name> -h`.

## Pipeline Overview

![Pipeline Diagram](images/videopipeline.png)

**Figure 2:** KABR tools pipeline for processing drone videos. The pipeline consists of four main steps: video data collection, data pre-processing, behavior labeling, and ecological analysis. Each step is modular and can be adapted to different species and study systems.

## Getting Started

To get started with KABR tools, follow the pipeline steps in order:

1. **[Data Collection](pipeline/data-collection.md)** - Collect drone video data following best practices
2. **[Pre-processing](pipeline/preprocessing.md)** - Use CVAT to create mini-scenes from your videos  
3. **[Behavior Labeling](pipeline/behavior-labeling.md)** - Apply machine learning models to classify behaviors
4. **[Analysis](pipeline/analysis.md)** - Generate ecological insights and visualizations

## Additional Resources

- [KABR Project Page](https://imageomics.github.io/KABR/) for additional details on the dataset and original paper.
- [KABR Mini-Scene Dataset on Hugging Face](https://huggingface.co/datasets/imageomics/KABR)
- [Pre-trained Model](https://huggingface.co/imageomics/x3d-kabr-kinetics)
- [KABR Collection on Hugging Face](https://huggingface.co/collections/imageomics/kabr-664dff304d29e6cd7b8e1a00): All datasets and models associated to the KABR Project.

## Citation

If you use KABR tools in your research, please follow the [citation guidance in the repo](https://github.com/Imageomics/kabr-tools?tab=readme-ov-file#citation).
