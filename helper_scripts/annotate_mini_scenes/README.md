The scripts provided in the /helper_scripts/annotate_mini_scenes directory allow users to submit jobs to a SLURM scheduler.\
The jobs are submitted to automatically annotate mini-scene videos with behavior labels using the provided model (defined in `run.sh`).\
\
To use the scripts:
1. Update jobs and account number in `launch_jobs.py`
2. Run `launch_jobs.py`. This will submit `run.sh`, which calls `miniscene2behavior` with job parameters.
3. This will generate a csv file for each video with the following columns: [video, track, frame, label]. See /data/mini_scene_behavior_annotations in HuggingFace for example output.
