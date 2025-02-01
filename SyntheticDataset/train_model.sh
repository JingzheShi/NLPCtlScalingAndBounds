#!/bin/bash

# Arrays of parameters
context_lengths=(27 32 36 43 63 94)
dataset_sizes=(85000)
#(50000 75000 100000 150000 200000 400000 2000000)
weight_decays=(0.0008)

# Function to count running jobs for each GPU
count_running_jobs() {
    local device=$1
    ps aux | grep "train_model.py" | grep "cuda:$device" | grep -v grep | wc -l
}

# Function to run a single training job
run_training() {
    local ctl=$1
    local size=$2
    local decay=$3
    local device=$4
    
    # Format the filename with parameters
    local filename="011051_ctl${ctl}_dtsz${size}_wd${decay/./-}_cuda${device}.txt"
    
    echo "Starting training with CTL=$ctl, Dataset Size=$size, Weight Decay=$decay on cuda:$device"
    nohup python3 train_model.py \
        --context_length "$ctl" \
        --train_dataset_size "$size" \
        --weight_decay "$decay" \
        --device "cuda:$device" >> "$filename" &
}

# Main loop
for ctl in "${context_lengths[@]}"; do
    for size in "${dataset_sizes[@]}"; do
        for decay in "${weight_decays[@]}"; do
            # Wait until at least one GPU is free
            while true; do
                cuda0_jobs=$(count_running_jobs 0)
                cuda1_jobs=$(count_running_jobs 1)
                
                if [ $cuda0_jobs -lt 2 ]; then
                    run_training "$ctl" "$size" "$decay" "0"
                    break
                elif [ $cuda1_jobs -lt 2 ]; then
                    run_training "$ctl" "$size" "$decay" "1"
                    break
                else
                    echo "Both GPUs are busy, waiting..."
                    sleep 60
                fi
            done
            
            # Small delay to prevent race conditions
            sleep 10
        done
    done
done

echo "All training jobs have been queued!"

# Optional: Wait for all jobs to complete
wait
echo "All training jobs have completed!"