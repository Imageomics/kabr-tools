# Optional Steps

## Fine-tune YOLO for Your Dataset

If you wish to use YOLO to automatically generate detections, you may want to fine-tune your YOLO model for your dataset using the [YOLO training notebook](../notebooks/train_yolo.ipynb).

### cvat2ultralytics

Convert CVAT annotations to Ultralytics YOLO dataset format for training custom models.

**Usage:**
```bash
cvat2ultralytics --video path_to_videos --annotation path_to_annotations --dataset dataset_name [--skip skip_frames]
```

**Source:** [src/kabr_tools/cvat2ultralytics.py](../../src/kabr_tools/cvat2ultralytics.py)

## Additional Utility Tools

### player

Interactive player for tracking and behavior observation with visualization capabilities.

**Usage:**
```bash
player --folder path_to_folder [--save] [--imshow]
```

![Player Output](../images/playeroutput.png)

**Figure:** Example player.py output showing tracking visualization.

**Source:** [src/kabr_tools/player.py](../../src/kabr_tools/player.py)

### cvat2slowfast

Convert CVAT annotations to the dataset in Charades format for use with SlowFast video understanding models.

**Usage:**
```bash
cvat2slowfast --miniscene path_to_mini_scenes --dataset dataset_name --classes path_to_classes_json [--old2new path_to_old2new_json] [--no_images]
```

**Source:** [src/kabr_tools/cvat2slowfast.py](../../src/kabr_tools/cvat2slowfast.py)

## Helper Scripts

Several utility scripts are available in the `helper_scripts` directory:

### Annotate Mini-scenes

Scripts for batch processing and annotation of mini-scenes:

- **[launch_job.py](../../helper_scripts/annotate_mini_scenes/launch_job.py)** - Job launcher for batch processing.
- **[run.sh](../../helper_scripts/annotate_mini_scenes/run.sh)** - Shell script for running annotation tasks.

See [README](../../helper_scripts/annotate_mini_scenes/README.md) for detailed usage instructions.

### Video Processing Utilities

- **[downgrade.sh](../../helper_scripts/downgrade.sh)** - Reduce video file sizes for CVAT compatibility.
- **[rename.sh](../../helper_scripts/rename.sh)** - Batch rename files with consistent naming conventions.

## Advanced Configuration

### Custom Model Training

For training custom behavior recognition models, you'll need:

1. **Annotated mini-scenes** from your specific species/environment.
2. **Behavior ethogram** defining your classification categories.
3. **Training configuration** adapted to your dataset size and complexity.

### GPU Requirements

Most tools support both CPU and GPU processing:

- **CPU mode**: Slower but works on any system.
- **GPU mode**: Significantly faster, recommended for large datasets.
- **Multi-GPU**: Supported for distributed processing of large video collections.

### Batch Processing

For processing large video collections:

1. Use the provided shell scripts for batch operations.
2. Consider using job scheduling systems (SLURM, PBS) for cluster environments.
3. Monitor disk space requirements for intermediate files.

## Customization Options

The modular design allows for customization at multiple levels:

- **Detection models**: Swap YOLO versions or use custom trained models.
- **Tracking algorithms**: Modify tracking parameters for different scenarios.
- **Behavior models**: Train species-specific or behavior-specific classifiers.
- **Analysis pipelines**: Adapt visualization and metrics to research questions.

## Performance Optimization

Tips for optimizing performance:

- Use appropriate video resolution for your analysis needs.
- Batch process multiple videos simultaneously when possible.
- Utilize GPU acceleration for model inference.
- Consider video compression for storage efficiency.