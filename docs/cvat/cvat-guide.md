# CVAT Setup and Usage Guide

This guide provides comprehensive instructions for setting up and using CVAT (Computer Vision Annotation Tool) for video annotation tasks in the KABR tools pipeline.

## Overview

We chose to set up a self-hosted instance of CVAT on a remote server. We chose this option so we could manage our data more easily, and every person on our team could access the server remotely to complete their annotations.

We set up our instance of CVAT on a server with a AMD EPYC 7513 32-Core Processor, running Ubuntu 20.04.6 LTS (GNU/Linux 5.4.0-190-generic x86_64).

!!! note "Alternative Installation Options"
    CVAT may also be accessed online using CVAT Cloud or installed locally on Microsoft, Apple, and Linux computers. See the CVAT [Getting Started](https://docs.cvat.ai/docs/getting_started/overview/) documentation for instructions.

## Setting up CVAT

Please have a look at the [CVAT Installation Guide](https://docs.cvat.ai/docs/administration/basics/installation/#ubuntu-22042004-x86_64amd64) for instructions to set up the tool using Docker on a computer running Ubuntu OS.

See CVAT's [Manuals](https://docs.cvat.ai/docs/manual/) for further information.

### Detailed Installation Instructions

**Step 1: Retrieve the CVAT source code**

As of writing, the latest [release](https://github.com/cvat-ai/cvat/releases) of CVAT is v2.17.0. CVAT is updated frequently with new features and bug fixes. You may retrieve this (or any) specific version using `wget`:

```bash
CVAT_VERSION="v2.17.0" && CVAT_V="${CVAT_VERSION#v}"
wget https://github.com/cvat-ai/cvat/archive/refs/tags/${CVAT_VERSION}.zip && unzip ${CVAT_VERSION}.zip && mv cvat-${CVAT_V} cvat && rm ${CVAT_VERSION}.zip && cd cvat
```

**Step 2: Set the CVAT_HOST environment variable**

!!! warning "One-time Setup"
    This should only be done once. Skip this if upgrading.

```bash
echo "export CVAT_HOST=localhost" >> ~/.bashrc
source ~/.bashrc
```

**Step 3: Optionally mount a host volume**

To access data within CVAT from the host machine (rather than needing to upload through the browser), create the file `docker-compose.override.yml` in the same directory as `docker-compose.yml`. Add the following to the `docker-compose.override.yml` file:

```yaml
services:
  cvat_server:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_import:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_export:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_annotation:
    volumes:
      - cvat_share:/home/django/share:ro

volumes:
  cvat_share:
    driver_opts:
      type: none
      device: /abs/path/to/host/data/directory # Edit this line
      o: bind
```

**Step 4: Build CVAT**

!!! note "Host Volume"
    Exclude the `-f docker-compose.override.yml` below if not mounting a host volume.

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up --build -d
```

## Accessing CVAT

After CVAT was set up on our remote server, users ran the following commands in their terminal to use the CVAT web interface:

```bash
ssh username@servername -N -L 8080:localhost:8080
```

Next, users navigated to Chrome, and entered `http://localhost:8080/` into their browser to open the CVAT GUI.

![CVAT Interface](https://github.com/user-attachments/assets/0176b8f6-ca5d-416e-8ba8-a71ddc3d93ae)

## Creating Tasks in CVAT

1. **Open CVAT web GUI** by navigating to `http://localhost:8080/` in Chrome.

1. **Login to CVAT.**

1. **Navigate to Projects** and click on your project.

1. **Click on 'Create multi tasks'**

    ![Create Tasks](https://github.com/user-attachments/assets/bc53acb2-3a26-443a-9fd9-97d79bb0cddf)

1. **Select files** - Navigate to 'Connected file share' and select the files you want to upload to create tasks.

    !!! warning "Processing Time"
        This may take some time if the files are large.

    ![File Selection](https://github.com/user-attachments/assets/eb07f93d-454b-4fad-b19e-ccc09b133930)

1. **Clean up** - Once a task is created for the video, you may delete the original video file from the server since CVAT will save the data in a different location.

## Detections in CVAT

### Manual Bounding Box Detection

1. **Access your task** - Once you login to CVAT, navigate to the Projects tab where you should see your assigned tasks. Click on the task to open it.

1. **Assign and open task** - Choose a task to work and assign it to yourself under 'Assignee'. Click on 'Job #' to open task.

    ![Task Assignment](https://github.com/user-attachments/assets/0bcb1246-1bcf-40e0-be23-2492cd3546ef)

1. **Optional: Select region of interest** - Select a region of interest to zoom in on the scene.

    ![Region Selection](https://github.com/user-attachments/assets/981fa973-7c59-4f2b-82c6-7a583ff2b656)

1. **Set up detection tool** - Click on the rectangle and select the correct species. Check the filename if you aren't sure. Make sure to select "Track".

    ![Detection Setup](https://github.com/user-attachments/assets/74c54802-f9e8-4db3-95e8-1260ef866774)

1. **Draw bounding boxes** - Draw a box around each animal in view.

    ![Bounding Boxes](https://github.com/user-attachments/assets/77c6da7c-c989-4060-b0f7-d3db33d455f8)

1. **Track through frames** - Use the >> button, or press 'V' on your keyboard to advance 10 frames. Update the bounding boxes by dragging them to the correct positions as required.

    !!! important "Save Frequently"
        Make sure to save frequently! You can stop, save, and start working on the video later at any time.

    ![Frame Navigation](https://github.com/user-attachments/assets/1a8c9432-463e-417a-aea9-55152d206f58)

1. **Complete annotation** - Continue until all the frames have been annotated. Save your results.

1. **Mark as complete** - Once you are done, select "Validation" so the project lead knows the task is complete.

    ![Task Completion](https://github.com/user-attachments/assets/877a6f65-8766-4967-aa28-90f6943f3641)

## Behavior Labeling in CVAT

1. **Access your task** - Once you login to CVAT, navigate to the Projects tab where you should see your assigned tasks. Click on the task to open it.

    ![Behavior Task](https://github.com/user-attachments/assets/e6859ef0-e5b2-4024-9728-3fa4b86cfa6d)

1. **Set up point annotation** - Click on "Draw new points" (1) then select the appropriate animal (2). Next, select "Track".

    ![Point Setup 1](https://github.com/user-attachments/assets/5d9d4ef2-1eba-4141-8f0b-3dcd7ad1a91a)
    ![Point Setup 2](https://github.com/user-attachments/assets/d83ba1cd-7894-4b1d-99bc-63afa9b84209)

1. **Annotate behavior** - Place your dot anywhere on the screen and select the appropriate behavior. Here, the zebra is "Head Up".

    ![Behavior Selection](https://github.com/user-attachments/assets/caa7bb3b-c27d-4e6d-85d2-86adee547750)

1. **Continue annotation** - Continue annotating the video, updating the object label as the animal changes behavior. See the Updated Ethogram for explanations of the different behavior categories. Pay particular attention to the caveats on any "out of sight" sub-categories.

    !!! tip "Save Frequently"
        Make sure to save current changes frequently. You can annotate part of a video and come back to complete it later.

1. **Complete the job** - Once you are done annotating, save the annotations for a final time, change the job status to "completed" and then select "Finish the job."

    ![Job Completion](https://github.com/user-attachments/assets/d6406758-75a0-4a99-a588-8fab66ff6127)

## Downloading Annotations

You may download annotations for individual tasks, or the entire project.

1. **Access export options** - Click on the 3 vertical dots in the lower right corner of the project or task.

1. **Select export type** - Select "Export dataset" to export a project, or "Export task dataset" to export a task.

    ![Export Options 1](https://github.com/user-attachments/assets/d22d3265-9e26-4bd8-81de-403eb4215d1f)
    ![Export Options 2](https://github.com/user-attachments/assets/16a0bf90-9dab-4e70-8c4b-d976a5f79a04)

1. **Configure export** - Export in "CVAT for video 1.1" format, deselect the option to "Save images", and click "Ok".

    ![Export Configuration](https://github.com/user-attachments/assets/7678c340-99d4-437f-a153-f220e9149ddc)

1. **Download dataset** - Once the export request is complete, navigate to the "Requests" tab and download the dataset.

    ![Download Dataset](https://github.com/user-attachments/assets/ba911424-9972-4720-8f0e-8713c4645b4c)

### Using Your Annotations

**For detections:** You may use your detections to create mini-scenes using [tracks_extractor](../pipeline/preprocessing.md#step-2b-create-mini-scenes-from-tracks). These mini-scenes can be automatically annotated with behavior labels using the [KABR model](https://huggingface.co/imageomics/x3d-kabr-kinetics).

**For behaviors:** You may use your annotated behaviors to fine-tune a behavior recognition model, such as X3D, or create time-budgets with the [time budget notebook](../case_studies/0_time_budget/time_budget.ipynb).
