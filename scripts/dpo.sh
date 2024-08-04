set -e
set -x

export OMP_NUM_THREADS=2

log_file="dpo-ref-free"

config_file="recipes/pure-policy/dpo/config_full.yaml"

echo "logging to $log_file.log"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 2930 \
    sppo/run_dpo.py "$config_file" 
    # 2>&1 | tee "${log_file}.log"
