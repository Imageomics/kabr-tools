# Step 3: Label Mini-scenes with Behavior

## Overview

You can use the [KABR model](https://huggingface.co/imageomics/x3d-kabr-kinetics) on Hugging Face to label the mini-scenes with behavior. See the [ethogram](https://github.com/Imageomics/kabr-tools/tree/main/CVAT/ethogram) folder for the list of behaviors used to label the zebra videos.

## Using the miniscene2behavior Tool

Label the mini-scenes using the following command:

```bash
miniscene2behavior [--hub huggingface_hub] [--config path_to_config] --checkpoint path_to_checkpoint [--gpu_num number_of_gpus] --miniscene path_to_miniscene [--output path_to_output_csv]
```

## Usage Examples

### Download checkpoint from Hugging Face and extract config

```bash
miniscene2behavior --hub imageomics/x3d-kabr-kinetics --checkpoint checkpoint_epoch_00075.pyth.zip --miniscene path_to_miniscene
```

### Download checkpoint and config from Hugging Face

```bash
miniscene2behavior --hub imageomics/x3d-kabr-kinetics --config config.yml --checkpoint checkpoint_epoch_00075.pyth --miniscene path_to_miniscene
```

### Use local checkpoint and config

```bash
miniscene2behavior --config config.yml --checkpoint checkpoint_epoch_00075.pyth --miniscene path_to_miniscene
```

## Important Notes

!!! note "GPU Usage"
    If `gpu_num` is 0, the model will use CPU. Using at least 1 GPU greatly increases inference speed. If you're using OSC, you can request a node with one GPU by running:
    ```bash
    sbatch -N 1 --gpus-per-node 1 -A [account] --time=[minutes] [bash script]
    ```

!!! info "Input Format"
    Mini-scenes are clipped videos focused on individual animals and video is the raw video file from which mini-scenes have been extracted.

## Resources

- [Pre-trained KABR model](https://huggingface.co/imageomics/x3d-kabr-kinetics) on Hugging Face.
- [Ethogram definitions](https://github.com/Imageomics/kabr-tools/tree/main/CVAT/ethogram) - Behavior classification system used for zebra videos.
- [Example annotated outputs](https://huggingface.co/imageomics/x3d-kabr-kinetics/tree/main/data/mini_scene_behavior_annotations) on Hugging Face.

## Tool Reference

### miniscene2behavior

Source: [src/kabr_tools/miniscene2behavior.py](https://github.com/Imageomics/kabr-tools/blob/master/src/kabr_tools/miniscene2behavior.py)

Apply machine learning models to classify animal behaviors from mini-scene videos.

**Parameters:**
- `--hub`: Hugging Face hub repository containing model files.
- `--config`: Path to configuration file (local or from hub).
- `--checkpoint`: Path to model checkpoint file.
- `--gpu_num`: Number of GPUs to use (0 for CPU).
- `--miniscene`: Path to mini-scene videos directory.
- `--output`: Path for output CSV file (optional).

## Next Steps

Once you have labeled your mini-scenes with behaviors, proceed to [Step 4: Ecological Analysis](analysis.md) to generate insights and visualizations from your behavioral data.
