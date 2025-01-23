set -e
set -x

# Default num GPUs
NUM_GPUS=8

# Function to generate comma-separated list of GPU IDs
generate_gpu_list() {
    local num=$1
    local list=""
    for ((i=0; i<num; i++)); do
        if [ $i -eq 0 ]; then
            list="$i"
        else
            list="$list,$i"
        fi
    done
    echo "$list"
}

HF_ORG=UCLA-AGI

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTDIR="data-mistral-7b-instruct-sppo-iter1"

PAIRS=5
FRAC=0
PROMPTS="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --pairs)
        PAIRS="$2"
        shift
        ;;
    --frac)
        FRAC="$2"
        shift
        ;;
    --model)
        MODEL="$2"
        shift
        ;;
    --out_path)
        OUTDIR="$2"
        shift
        ;;
    --prompt)
        PROMPTS="$2"
        shift
        ;;
    --gpus)
        NUM_GPUS="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

# Validate number of GPUs
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: Number of GPUs must be at least 1"
    exit 1
fi

# Generate GPU list and array
export CUDA_VISIBLE_DEVICES=$(generate_gpu_list $NUM_GPUS)
AVAILABLE_GPUS=($(seq 0 $((NUM_GPUS-1))))

echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

#####################
# Generate Data
#####################

#frac length 2600 * num_gpus 8 = 20800, should be larger than the length of the dataset. Change frac_len accordingly when dataset changes

FRAC_LEN=$((20800 / ${#AVAILABLE_GPUS[@]}))
echo "Using frac_len ${FRAC_LEN}"
(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/generate.py --model $MODEL --maxlen 2048 --output_dir "generated/$OUTDIR" --prompts $PROMPTS --pairs $PAIRS --world_size 1 --frac_len $FRAC_LEN --data_frac $data_frac > output_log_${gpu_id}.txt 2>&1 &
        ((data_frac+=1));
    done
    wait
) &
all_gen=$!

wait $all_gen

python3 scripts/combine_generate.py --output_dir "generated/$OUTDIR" --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" --pairs $PAIRS


# #####################
# # Rank Data
# #####################

# # frac length 2600 * num_gpus 8 = 20800, should be larger than the length of the dataset. Change frac_len accordingly when dataset changes

python3 scripts/preload.py

(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/rank.py --model $MODEL --output_dir $OUTDIR --pairs $PAIRS --numgpu ${#AVAILABLE_GPUS[@]} --frac_len $FRAC_LEN --data_frac $data_frac --gpu $gpu_id --prompts $PROMPTS > rank_log_${gpu_id}.txt 2>&1 &
        ((data_frac+=1));
    done
    wait
) &
all_rank=$!

wait $all_rank

python3 scripts/compute_prob.py --org $HF_ORG --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" --output_dir $OUTDIR --pairs $PAIRS --frac_len $FRAC_LEN --prompts $PROMPTS
