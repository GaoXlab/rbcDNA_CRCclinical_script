#!/bin/bash

TARGET_DIR="$(pwd)/figure_scripts/FeatureAnno/refData"

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR"

curl -L https://raw.githubusercontent.com/ernstlab/full_stack_ChromHMM_annotations/main/state_annotations_processed.csv > state_annotations_processed.csv
python3 -c "
import csv, sys
with open('state_annotations_processed.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 5:
            print(f'{row[0]}\t{row[1]}\t{row[4]}')
" > state_annotations_processed

wget "https://public.hoffman2.idre.ucla.edu/ernst/UUKP7/hg38lift_genome_100_segments.bed.gz"

gzip -d hg38lift_genome_100_segments.bed.gz