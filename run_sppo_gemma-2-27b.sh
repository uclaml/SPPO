#!/bin/bash
iter_num=3
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="google/gemma-2-27b-it"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Gemma-2-27B-SPPO-It-Iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"
    OUT="data-gemma-2-27b-it-sppo-iter${i}"
    echo "runing epoch $i"
    DATASET="synthetic_data_gemma-2-27b-it-sppo-iter${i}_score"

    if [ ! -d "$DATASET" ]; then
        bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    fi
    bash scripts/pipeline.sh --model $MODEL --iter $i --dataset "$DATASET" --output_dir $OUTPUT_DIR --num 1 --batch_size 1 --accumulate 8
done
