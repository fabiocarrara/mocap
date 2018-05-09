#!/bin/bash

# FPSs=60 30 15 10
# FPSs=40 24 20 12 8 6
# FPSs=1 2 4 0.5 6 8 12 20 24 40
# FPSs=120 60 40 24 20 12 8 6 4 2 1 0.5

FPSs=120

for FPS in $FPSs; do
for DIR in '-u' '-b'; do

# HDM05-122
python train_segment.py --run-dir runs_segmentation_hdm05-122 \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-1-of-2.pkl \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-2-of-2.pkl \
    $DIR -f $FPS

python train_segment.py --run-dir runs_segmentation_hdm05-122 \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-2-of-2.pkl \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-1-of-2.pkl \
    $DIR -f $FPS

# HDM05-65
python train_segment.py --run-dir runs_segmentation_hdm05-65 \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-1-of-2.pkl \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-2-of-2.pkl \
    --mapper data/HDM05-category_mapper-130vs65.csv \
    $DIR -f $FPS

python train_segment.py --run-dir runs_segmentation_hdm05-65 \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-2-of-2.pkl \
    data/HDM05-122/HDM05-122-whole-seq+annot-fold-1-of-2.pkl \
    --mapper data/HDM05-category_mapper-130vs65.csv \
    $DIR -f $FPS

# HDM05-15
python train_segment.py --run-dir runs_segmentation_hdm05-15 \
    data/HDM05-15/HDM05-15-whole-seq+annot-fold-1-of-2.pkl \
    data/HDM05-15/HDM05-15-whole-seq+annot-fold-2-of-2.pkl \
    $DIR -f $FPS

python train_segment.py --run-dir runs_segmentation_hdm05-15 \
    data/HDM05-15/HDM05-15-whole-seq+annot-fold-2-of-2.pkl \
    data/HDM05-15/HDM05-15-whole-seq+annot-fold-1-of-2.pkl \
    $DIR -f $FPS

# HDM05-15 20/80 split
python train_segment.py --run-dir runs_segmentation_hdm05-15_20-80 \
    data/HDM05-15/HDM05-15-whole-seq+annot-split-20-80-train.pkl \
    data/HDM05-15/HDM05-15-whole-seq+annot-split-20-80-test.pkl \
    $DIR -f $FPS

done
done

