# --------------
# CLASSIFICATION
# --------------

## HDM05-122, 2-FOLD SPLITS (SAME FOR HDM05-65)
pushd data/HDM05-122/
python ../../parse_data.py \
    HDM05-122-objects-annotations-specific-coords_normPOS.data \
    -s splits/annotated_subsequence_splits/2_folds/HDM05-122-only-annot-subseq-2folds_1-cats122.txt --sf list \
    HDM05-122-only-annot-subseq-fold-1-of-2.pkl

python ../../parse_data.py \
    HDM05-122-objects-annotations-specific-coords_normPOS.data \
    -s splits/annotated_subsequence_splits/2_folds/HDM05-122-only-annot-subseq-2folds_2-cats122.txt --sf list \
    HDM05-122-only-annot-subseq-fold-2-of-2.pkl

## HDM05-122, 10-FOLD SPLITS (SAME FOR HDM05-65)

for I in {1..10}; do
for MODE in 'train' 'test'; do

    python ../../parse_data.py \
        HDM05-122-objects-annotations-specific-coords_normPOS.data \
        -s splits/annotated_subsequence_splits/10_folds/HDM05-122-only-annot-subseq-10fold_${I}_${MODE}.txt --sf list \
        HDM05-122-only-annot-subseq-fold-${I}-of-10-${MODE}.pkl

done
done

popd

# ------------
# SEGMENTATION
# ------------

## HDM05-122, 2 SPLITS (SAME FOR HDM05-65)
pushd data/HDM05-122/

python ../../parse_data.py \
    HDM05-122-objects-annotations-specific-coords_normPOS.data \
    HDM05-122-objects-annotations-specific-coords_normPOS.pkl


python ../../parse_data.py \
    HDM05-122-objects-sequences_annotated-coords_normPOS.data \
    -a split/whole_sequence_split/HDM05-122-whole-seq+annot-2fold.txt --af petr \
    -s split/whole_sequence_split/HDM05-122-whole-seq+annot-2fold.txt --sf petr \
    HDM05-122-whole-seq+annot-fold-1-of-2.pkl

python ../../parse_data.py \
    HDM05-122-objects-sequences_annotated-coords_normPOS.data \
    -a split/whole_sequence_split/HDM05-122-whole-seq+annot-2fold.txt --af petr \
    -s split/whole_sequence_split/HDM05-122-whole-seq+annot-2fold.txt --sf petr \
    HDM05-122-whole-seq+annot-fold-2-of-2.pkl --test

popd

## HDM05-15
pushd data/HDM05-15/

python ../../parse_data.py \
    HDM05-15-objects-annotations-specific-coords_normPOS.data \
    HDM05-15-objects-annotations-specific-coords_normPOS.pkl

### 50-50 SPLITS (2 FOLD)
python ../../parse_data.py \
    HDM05-15-objects-sequences_annotated-coords_normPOS.data \
    -a split/whole_sequence_split/HDM05-15-whole-seq+annot-2fold.txt --af petr \
    -s split/whole_sequence_split/HDM05-15-whole-seq+annot-2fold.txt --sf petr \
    HDM05-15-whole-seq+annot-fold-1-of-2.pkl

python ../../parse_data.py \
    HDM05-15-objects-sequences_annotated-coords_normPOS.data \
    -a split/whole_sequence_split/HDM05-15-whole-seq+annot-2fold.txt --af petr \
    -s split/whole_sequence_split/HDM05-15-whole-seq+annot-2fold.txt --sf petr \
    HDM05-15-whole-seq+annot-fold-2-of-2.pkl --test

### 20-80 SPLITS
python ../../parse_data.py \
    HDM05-15-objects-sequences_annotated-coords_normPOS.data \
    -a HDM05-15-objects-annotations-coords_normPOS.pkl --af pkl \
    -s splits/whole_sequence_splits/HDM05-15-whole-seq+annot-split-20-80-train.txt --sf csv \
    HDM05-15-whole-seq+annot-split-20-80-train.pkl

python ../../parse_data.py \
    HDM05-15-objects-sequences_annotated-coords_normPOS.data \
    -a HDM05-15-objects-annotations-coords_normPOS.pkl --af pkl \
    -s splits/whole_sequence_splits/HDM05-15-whole-seq+annot-split-20-80-test.txt --sf csv \
    HDM05-15-whole-seq+annot-split-20-80-test.pkl

popd