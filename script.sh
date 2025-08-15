#!/bin/bash

# Create output directory
mkdir -p output
mkdir -p label
# Compile using Makefile
# make

# Loop over each input file in the input/ directory
for input_file in input/*.txt; do
    filename=$(basename "$input_file")               # Extract filename (e.g., input_0.txt)
    output_file="output/$filename"                   # Corresponding output file path
    log_msg="Running test on $filename"

    echo "$log_msg"

    {
        echo "$log_msg"
        echo "[CUDA Output]"
        ./main "label/$filename" < "$input_file"
        echo "================"
        echo "[Python Output]"
        python3 km.py "$input_file"
    } > "$output_file"
done

echo "All tests completed. Outputs saved in output/"

