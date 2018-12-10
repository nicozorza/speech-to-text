#!/bin/bash
# This scripts takes all files in the TIMIT database and copies them to another folder
search_dir=/home/nicozorza/Escritorio/speech-to-text/data/wav/TIMIT
results_dir=results

mkdir "$results_dir"
mkdir "$results_dir/TRAIN"
mkdir "$results_dir/TEST"

for data_dir in TEST TRAIN
do
	for d in $(find "$search_dir/$data_dir/" -maxdepth 2 -type d)
	do
		for file in "$d"/*
		do
			extension="${file##*.}"
			filename="${file%.*}"
			# El -f11 depende del path search_dir (en algun momento lo automatizare)
			folder="$(cut -d'/' -f11 <<<${filename})"

			cp "$file" "${filename}_${folder}.${extension}"
			mv "${filename}_${folder}.${extension}" "$results_dir/$data_dir/"
			echo "$0: '${file}'"
		done
	done
done
