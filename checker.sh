#!/bin/bash

LABEL_DIR1="./label"
LABEL_DIR2="./cuML/labels"
OUTPUT_FILE="comparison_results.txt"
COMPARE_SCRIPT="checker.py"

# Empty the output file if it exists
> "$OUTPUT_FILE"

for file1 in "$LABEL_DIR1"/*; do
    filename=$(basename "$file1")
    file2="$LABEL_DIR2/$filename"

    if [[ ! -f "$file2" ]]; then
        echo "Skipping $filename: file not found in $LABEL_DIR2" >> "$OUTPUT_FILE"
        continue
    fi

    {
        echo "========================================"
        echo "Comparing: $file1 vs $file2"
        echo "----------------------------------------"
        python "$COMPARE_SCRIPT" "$file1" "$file2"
        echo
    } >> "$OUTPUT_FILE"
done

