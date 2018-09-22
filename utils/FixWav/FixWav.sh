#!/bin/bash
# This scripts allows to convert the Sphere NIST WAV format to normal WAV format

if [ "$#" -eq 1 ]
then
	search_dir=$1
	this_dir=FixWav
	echo $this_dir

	cd ..
	for file in "$search_dir"/*
	do
	  	echo "$file"
		./"$this_dir"/sph2pipe -t : -f rif "$file" out.wav
		mv out.wav "$file"
	done
else
	echo "Not enough arguments"
	echo "Usage: ./FixWav.sh <source_dir>"
	exit 1
fi
