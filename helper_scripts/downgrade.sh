# reduce the quality of all mp4 files in a directory
# usage: ./downgrade.sh /path/to/videos

# get the path to the videos
path=$1

# get the videos in the path
videos=$(ls $path)

for video in $videos
do
    # downgrade the video
    ffmpeg -i $path/$video -vf scale=1824:1026 $path/downgraded-$video
done

# original size: 5472 x 3078
# 1/2 size: 2736 x 1539
# 1/3 size: 1824 x 1026
# 1/4 size: 1368 x 769
# 1/8 size: 684 x 384 (tried 8.55 which is 640:360, but too blurry)