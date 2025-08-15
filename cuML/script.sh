#!/bin/bash

# Create output directories
mkdir -p output
mkdir -p labels

# Compile using Makefile (if needed)
# make

# Loop over each input file in the input/ directory
for input_file in input/*.txt; do
    filename=$(basename "$input_file")             # Extract filename (e.g., input_0.txt)
    output_file="output/$filename"                 # File to capture console output
    label_file="labels/$filename"                  # File to save cluster labels
    log_msg="Running test on $filename"

    echo "$log_msg"

    {
        echo "$log_msg"
        echo "================"
        echo "[Python Output]"
        python3 km_cu.py "$input_file" --output_file "$label_file"
    } > "$output_file"
done

echo "All tests completed. Outputs saved in output/"
echo "Cluster labels saved in labels/"

