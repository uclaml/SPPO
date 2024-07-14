set -e
set -x

export OMP_NUM_THREADS=2

LEARNING_RATE="5.0e-7"
ITER="1"
BETA="0.001"
LOSS_TYPE="sppo"
OPTIM="rmsprop"
PREF="sppo_score"
NUM=18
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DATASET="synthetic_data_mistral-7b-instruct-sppo-iter1_score"
BATCH_SIZE=8
ACCUMULATE=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --learning_rate)
        LEARNING_RATE="$2"
        shift
        ;;
    --beta)
        BETA="$2"
        shift
        ;;
    --optim)
        OPTIM="$2"
        shift
        ;;
    --output_dir)
        OUTPUT_DIR="$2"
        shift
        ;;
    --iter)
        ITER="$2"
        shift
        ;;
    --loss_type)
        LOSS_TYPE="$2"
        shift
        ;;
    --prefix)
        PREF="$2"
        shift
        ;;
    --model)
        MODEL="$2"
        shift
        ;;
    --dataset)
        DATASET="$2"
        shift
        ;;
    --num)
        NUM="$2"
        shift
        ;;
    --batch_size)
        BATCH_SIZE="$2"
        shift
        ;;
    --accumulate)
        ACCUMULATE="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

PREF="${PREF}_${NUM}"

LEVEL1="iter${ITER}_${LEARNING_RATE}_beta${BETA}_${OPTIM}"
LEVEL2="${LOSS_TYPE}_${PREF}"

#OUTPUT_DIR="checkpoints/${LEVEL1}/${LEVEL2}"
log_file="iter${ITER}_${LEARNING_RATE}_${BETA}_${OPTIM}_${LOSS_TYPE}_${PREF}"

dataset_name=$(echo "$DATASET" | cut -d '/' -f2)
new_config_file="recipes/uclaml-sppo/config_full_${dataset_name}.yaml"

# Copy the original configuration file to the new one
cp recipes/uclaml-sppo/config_full.yaml "$new_config_file"

python3 scripts/update_dataset.py --dataset $DATASET --config "$new_config_file" >"$log_file.log"

echo "logging to $log_file.log"

# --main_process_port ${port} \

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 2930 \
    sppo/run_sppo.py "$new_config_file" \
    --learning_rate=$LEARNING_RATE \
    --beta=$BETA \
    --optim="$OPTIM" \
    --output_dir="$OUTPUT_DIR" \
    --run_name="sppo" \
    --loss_type=$LOSS_TYPE \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$ACCUMULATE \
    --model_name_or_path=$MODEL \
    --num_train_epochs=$NUM
# 2>&1 | tee "${log_file}.log"
