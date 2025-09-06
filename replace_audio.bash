#!/usr/bin/env bash

ffmpeg -i song.mp4 -i song_cleaned.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 song_cleaned.mp4

#to know the length
#ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 song_cleaned.mp4