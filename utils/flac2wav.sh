#!/bin/bash
# This scripts allows to convert the audio files from .flac to .wav

if [ "$#" -eq 1 ]
then
	search_dir=$1

	for file in "$search_dir"/*
	do
		filename=$(basename $file .flac)
		echo "$filename.wav"

		ffmpeg -i $file "$search_dir/$filename.wav"
		
	done
else
	echo "Not enough arguments"
	echo "Usage: ./flac2wav.sh <source_dir>"
	exit 1
fi
