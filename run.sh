#!/bin/bash

# Create output directory
mkdir -p outputNew

# List of N values
N_list=(100000 500000 800000 1000000 700000 2000 10000 2000000 400000 5000000)
# Loop 10 times
for i in {0..9}; do
    N=${N_list[$i]}
    d=$((RANDOM % 41 + 80))  # Random number between 80 and 120
    k=$((RANDOM % 41 + 80))  # Random number between 80 and 120

    log_msg="Running test $i with N=$N, d=$d, k=$k"
    echo "$log_msg"

    # Generate dataset
    python3 TestGenerator.py "$N" "$d" "$k"

    # Define output file
    output_file="outputNew/output_${i}.txt"

    # Write log and CUDA output
    {
        echo "$log_msg"
        echo "[CUDA Output]"
        ./main < dataset.txt
        echo "================"
        echo "[Python Output]"
        python3 km.py dataset.txt
    } > "$output_file"

done

echo "All tests completed. Outputs saved in outputDump/"

