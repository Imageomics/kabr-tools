import os
import sys
import subprocess
import json

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python counter.py path_to_folder")
        exit(0)
    elif len(sys.argv) == 2:
        path = sys.argv[1]

    for root, dirs, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1] == ".mp4":
                video_path = os.path.join(root, name)
                result = subprocess.check_output(
                    f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{video_path}"',
                    shell=True).decode()
                fields = json.loads(result)["streams"][0]
                duration = float(fields["duration"])
                print(video_path[2:], f"Length: {duration:.0f}sec")
