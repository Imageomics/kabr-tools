# rename video names with date and species from filepath
# Usage: ./rename.sh /path/to/videos

# get the path to the videos
path=$1

# get the date and species from the path
date=$(echo $path | cut -d'/' -f4)
species=$(echo $path | cut -d'/' -f5)

# get the videos in the path
videos=$(ls $path)

# rename the videos
for video in $videos
do
    # rename the video
    mv $path/$video $path/$date-$species-$video
done