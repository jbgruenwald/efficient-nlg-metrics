#!/bin/bash

set -e

mkdir -p cache

GPU=1
L1=en
#L2=ru
MODEL=xlm-roberta-base
FAST_ALIGN=./fast_align/build
OUTPUT_PATH=../../../../xmoverscore-mappings/xlmr
CACHE=./cache
for L2 in de cs fi ro ru tr zh; do
    PARA=WikiMatrix.$L2-$L1.2k

    # download and prep parallel data
    #curl -LO https://www.statmt.org/europarl/v7/$L2-$L1.tgz
    #tar -zxvf $L2-$L1.tgz -C $OUTPUT_PATH/
    #rm -f $L2-$L1.tgz
    python3 prep_parallel.py --bert_model $MODEL --lang1 ../../../../xmoverscore-mappings/data/WikiMatrix.$L1-$L2.$L2 --lang2 ../../../../xmoverscore-mappings/data/WikiMatrix.$L1-$L2.$L1 --size 2000 --output $OUTPUT_PATH/$PARA

    # get word alignment
    $FAST_ALIGN/fast_align -i $OUTPUT_PATH/$PARA.uncased -d -o -v > $OUTPUT_PATH/forward.align.$PARA.uncased
    $FAST_ALIGN/fast_align -i $OUTPUT_PATH/$PARA.uncased -d -o -v -r > $OUTPUT_PATH/reverse.align.$PARA.uncased
    $FAST_ALIGN/atools -i $OUTPUT_PATH/forward.align.$PARA.uncased -j $OUTPUT_PATH/reverse.align.$PARA.uncased -c grow-diag-final-and > $OUTPUT_PATH/sym.align.$PARA.uncased

    # produce alignment matrix (for the last 4 layers)
    for l in 12; do
        CUDA_VISIBLE_DEVICES=$GPU python3 get_aligned_features_avgbpe.py \
          --para_file $OUTPUT_PATH/$PARA.cased \
          --align_file $OUTPUT_PATH/sym.align.$PARA.uncased \
          --bert_model $MODEL \
          --cache_dir $CACHE \
          --layer $l \
          --output_file $OUTPUT_PATH/$PARA.$l

        python3 train_mapping.py --emb_file $OUTPUT_PATH/$PARA.$l --output_file $OUTPUT_PATH/$PARA.$l --orthogonal
    done
done

python3 convert-l8.py --directory=$OUTPUT_PATH