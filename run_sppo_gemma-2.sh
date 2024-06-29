#!/bin/bash
iter_num=3
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="google/gemma-2-9b-it"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Gemma-2-9B-SPPO-It-Iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"
    OUT="data-gemma-2-9b-it-sppo-iter${i}"
    echo "runing epoch $i"
    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline.sh --model $MODEL --iter $i --dataset "synthetic_data_gemma-2-9b-it-sppo-iter${i}_score" --output_dir $OUTPUT_DIR --num 1 --batch_size 4 --accumulate 2
done
