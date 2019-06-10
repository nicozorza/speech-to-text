#!/bin/bash
# This scripts takes all files in the Librispeech database and copies them to another folder

if [ "$#" -eq 1 ]
then
	search_dir=$1
	results_dir=$2
	for d in $(find "$search_dir/" -maxdepth 3 -type d)
	do
		for file in "$d"/*
		do
			extension="${file##*.}"
			if [ "$extension" == "flac" ]; then
				filename="${file##*/}"
	    			echo "${filename}"
				cp $file "$results_dir/$filename"
			fi
		done
	done
else
	echo "Not enough arguments"
	echo "Usage: ./CopyLibrispeech.sh <source_dir> <results_dir>"
	exit 1
fi

