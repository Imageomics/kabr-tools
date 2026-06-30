# This script is used to submit jobs to label mini_scenes to the SLURM scheduler.
import subprocess

account_number = 'account_number' # put your account number here

jobs = [
    {'miniscene':'',
     'count':0,
     'video':'',
     'output':''},
] # put jobs parameters to submit here

def submit_job(sbatch_args=None):
    if sbatch_args is None:
        sbatch_args = []
    
    # Prepare the command
    command = ["sbatch"] + sbatch_args
    print(' '.join(command))
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted successfully.")
            print(f"SLURM Job ID: {result.stdout.strip()}")
        else:
            print(f"Failed to submit.")
            print(f"Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"An error occurred while submitting: {e}")

for job in jobs:
    args = f"-N 1 -p gpuserial --gpus-per-node 1 -A {account_number} --time={30 * (job['count']+1)} run.sh {job['miniscene']} {job['video']} {job['output']}".split()
    submit_job(args)